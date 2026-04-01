import os
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

def augment_timeseries_multires(x, jitter_std=0.02, scaling_range=(0.9, 1.1)):
    """
    이미지 증강 기법의 시계열 이식 버전.
    Args:
        x: [C, T] 텐서
    """
    # 1. Random Horizontal Flip (Time Reversal) - 시계열 예측(Forecasting) 특성상 방향성 훼손으로 인해 제거됨
    # if random.random() < 0.5:
    #     x = torch.flip(x, dims=[-1])
        
    # 2. Gaussian Blur (Smoothing) - p=0.3
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        x = F.pad(x.unsqueeze(0), (kernel_size//2, kernel_size//2), mode='replicate')
        x = F.avg_pool1d(x, kernel_size=kernel_size, stride=1).squeeze(0)

    # 3. Jittering
    if jitter_std > 0:
        x = x + torch.randn_like(x) * jitter_std
    
    # 4. Scaling
    if scaling_range:
        scale = random.uniform(*scaling_range)
        x = x * scale
        
    return x

class MultiResolution1DTransform:
    def __init__(
        self, 
        global_len=512,
        local_len=256,
        n_global=2,
        n_local=6,
    ):
        self.global_len = global_len
        self.local_len = local_len
        self.n_global = n_global
        self.n_local = n_local

    def random_crop_1d(self, x, target_len):
        """1D 시계열에서 연속된 패치(Sub-sequence)를 랜덤 추출합니다.
        
        Returns:
            (cropped_tensor, start_offset): 잘라낸 텐서와 원본에서의 시작 타임스텝 인덱스
        """
        T = x.shape[-1]
        if T <= target_len:
            return x, 0  # 자를 필요 없을 때 offset=0
        start = random.randint(0, T - target_len)
        return x[..., start:start + target_len], start  # start offset 함께 반환

    def __call__(self, x):
        # Global View: 원본 시퀀스 길이 유지 (Sequence level)
        global_views = [augment_timeseries_multires(x) for _ in range(self.n_global)]
        
        # Local View: 증강 후 무작위 부분 구간(Patch level)으로 자름
        # random_crop_1d가 (tensor, start_offset) 튜플을 반환하므로 분리
        local_crops = [
            self.random_crop_1d(augment_timeseries_multires(x), self.local_len)
            for _ in range(self.n_local)
        ]
        local_views, local_offsets = zip(*local_crops)  # 텐서와 offset 분리
        
        return {
            'global': torch.stack(global_views),                              # [2, C, 512]
            'local': torch.stack(list(local_views)),                          # [6, C, 256]
            'local_offsets': torch.tensor(list(local_offsets), dtype=torch.long),  # [6] 타임스텝 단위
        }

class CSVMultiResDataset(Dataset):
    """기존 CSV2DDataset의 전처리 로직을 계승한 다중 해상도 데이터셋"""
    def __init__(self, csv_path, seq_len=512, stride=128, mode="train", transform=None):
        self.seq_len = seq_len
        self.stride = stride
        self.mode = mode
        self.transform = transform
        
        # 기존 data_2d.py의 로직 빌려오기
        df_np = self._load_and_preprocess(csv_path)
        total_len = len(df_np)
        val_len = int(total_len * 0.1)
        test_len = int(total_len * 0.2)
        train_len = total_len - val_len - test_len

        scaler = StandardScaler()
        scaler.fit(df_np[:train_len])
        df_norm = scaler.transform(df_np)

        if mode == "train": split_data = df_norm[:train_len]
        elif mode == "val": split_data = df_norm[train_len : train_len + val_len]
        else: split_data = df_norm[train_len + val_len :]

        self.data = torch.from_numpy(split_data).float()
        self.indices = [i for i in range(0, len(self.data) - self.seq_len + 1, self.stride)]
        print(f"✅ CSVMultiResDataset ({mode}): {len(self.indices)} 샘플 로드 완료")

    def _load_and_preprocess(self, path):
        # data_2d.py와 동일한 전처리 로직 (NaN 처리, 보간 등)
        df = pd.read_csv(path, low_memory=False)
        for c in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
            if c in df.columns: df.drop(columns=[c], inplace=True); break
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # TS-JEPA 특화 전처리: Weather 데이터셋 등에서 발생하는 극단 이상치(-9990 이하) 처리
        for col in numeric_cols:
            df.loc[df[col] < -9990.0, col] = np.nan
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df.dropna(inplace=True, axis=0)
        return df.values.astype(np.float32)

    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start : start + self.seq_len].T # [C, T]
        return self.transform(window) if self.transform else window

class TSLDMultiResDataset(Dataset):
    """TSLD 데이터셋 고속 로딩을 위한 Lazy Loading 구현"""
    def __init__(self, root_path, seq_len=512, stride=512, mode="train", max_files=None, transform=None):
        self.seq_len = seq_len
        self.stride = stride
        self.mode = mode
        self.transform = transform
        self.sample_indices = [] # (series_idx, start_idx, mean, std)
        self.time_series_list = [] # 전처리된 (T, 1) ndarray 저장용 리스트
        
        csv_files = sorted(
            [os.path.join(r, f) for r, _, fs in os.walk(root_path) for f in fs if f.endswith(".csv") and "hhh" not in f]
        )
        if max_files: csv_files = csv_files[:max_files]

        print(f"🔍 TSLD {mode} 인덱스 생성 중... (Lazy Loading)")
        for file_path in csv_files:
            try:
                # 데이터 전체를 읽지 않고 컬럼 정보와 스케일 통계량만 미리 계산
                df_meta = pd.read_csv(file_path, nrows=5) # 구조 파악용
                cols = [c for c in df_meta.columns if c not in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]]
                
                # 실제 계산을 위해 한 번은 전체 로드하되, 메모리에 유지하지 않음
                df = pd.read_csv(file_path, usecols=cols)
                # 이상치 처리 (TSL 방식: fillna로 속도 향상)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df))
                df = df.fillna(0)
                
                for col in cols:
                    # 1. 시계열 데이터 추출 및 기본 NaN 처리 (TS-JEPA 방식)
                    col_data = df[col].values.astype(np.float32)
                    col_data = pd.Series(col_data).ffill().fillna(0).values
                    
                    total_len = len(col_data)

                    # 2. 데이터가 최소 윈도우(seq_len)보다 짧으면 즉시 skip
                    if total_len < self.seq_len:
                        continue

                    # 3. Train/Val/Test 분할 지점 계산
                    val_len = int(total_len * 0.1)
                    test_len = int(total_len * 0.2)
                    train_len = total_len - val_len - test_len

                    # 4. Train 데이터가 없으면 통계량 계산이 불가능하므로 skip (에러 방지 핵심)
                    if train_len <= 0:
                        continue

                    # 5. 통계량 계산 (여기서 발생하는 Mean of empty slice 방지)
                    m = col_data[:train_len].mean()
                    s = col_data[:train_len].std() + 1e-8

                    # 6. Mode별 시작/끝 지점 결정
                    if mode == "train":
                        start_limit = train_len
                        offset = 0
                    elif mode == "val":
                        start_limit = train_len + val_len
                        offset = train_len
                    else: # test
                        start_limit = total_len
                        offset = train_len + val_len

                    current_len = start_limit - offset
                    if current_len >= self.seq_len:
                        # 11. 전처리 및 정규화 완료 후 메모리에 저장 (O(1) 접근용)
                        series_idx = len(self.time_series_list)
                        # Train 통계량(m, s)을 적용하여 미리 정규화
                        split_data = (col_data[offset:start_limit] - m) / s
                        split_data = split_data.reshape(-1, 1).astype(np.float32)
                        self.time_series_list.append(split_data)

                        for start in range(0, current_len - self.seq_len + 1, self.stride):
                            self.sample_indices.append((series_idx, start))
            except Exception as e:
                # 개별 파일 로드 실패 시 무시하고 진행
                # print(f"[ERROR] {file_path}: {e}")
                continue
        print(f"✅ TSLDMultiResDataset ({mode}): {len(self.sample_indices)} 샘플 인덱싱 완료")

    def __len__(self): return len(self.sample_indices)

    def __getitem__(self, idx):
        series_idx, start = self.sample_indices[idx]
        # 메모리 리스트에서 직접 인덱싱 (이미 정규화 완료됨)
        series_data = self.time_series_list[series_idx] # [T_split, 1]
        window = series_data[start : start + self.seq_len] # [seq_len, 1]
        
        # LeJEPA Expected Shape: [1, T]
        window = torch.from_numpy(window.T) 
        return self.transform(window) if self.transform else window

def get_1d_multires_loaders(dataset_type="electricity", path="", batch_size=64, seq_len=512, stride=128, num_workers=4, max_files=None, local_len=128, arch="basic"):
    # UTICA 아키텍처는 GPU에서 자체적으로 Crop을 수행하므로 Transform을 우회하여 원본 텐서를 그대로 반환합니다.
    transform = MultiResolution1DTransform(global_len=seq_len, local_len=local_len) if arch.lower() != "utica" else None
    if dataset_type == "tsld":
        # TSLD 표준: 항상 non-overlapping (stride = seq_len)
        train_ds = TSLDMultiResDataset(path, seq_len, stride, "train", max_files, transform)
        val_ds = TSLDMultiResDataset(path, seq_len, stride, "val", max_files, transform)
    else:
        train_ds = CSVMultiResDataset(path, seq_len, stride, "train", transform)
        val_ds = CSVMultiResDataset(path, seq_len, stride, "val", transform)
    return DataLoader(train_ds, batch_size, shuffle=True, drop_last=False, num_workers=num_workers), \
           DataLoader(val_ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
