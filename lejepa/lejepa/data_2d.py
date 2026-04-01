"""
TS_JEPA의 데이터 전처리 코드(Electricity, TSLD 등)를 이식하여
LeJEPA의 직사각형 2D ViT 입력 [3, H, W]에 맞게 반환해주는 데이터 로더.

H: 변수 개수(C)를 patch_size 배수로 올림
W: 시간 길이(seq_len, T)
"""

import os
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def augment_timeseries(x, noise_std=0.02, scale_range=(0.9, 1.1)):
    """
    Args:
        x: [C, T] 텐서
    Returns:
        Augmented [C, T] 텐서
    """
    # 1. Jittering (가우시안 노이즈)
    x_aug = x + torch.randn_like(x) * noise_std

    # 2. Scaling (채널별 랜덤 스케일링)
    C = x.shape[0]
    scale = (
        torch.rand(C, 1) * (scale_range[1] - scale_range[0])
        + scale_range[0]
    ).to(x.device)
    x_aug = x_aug * scale

    return x_aug


def ts_to_1ch_image(x_2d, patch_size=8):
    """
    Args:
        x_2d: [C, T] 연속 시계열
    Returns:
        [1, H, W] 2D Image (단일 채널)
        H = 패딩된 C, W = T
    """
    C, T = x_2d.shape
    H = math.ceil(C / patch_size) * patch_size
    W = T

    img = torch.zeros(1, H, W, dtype=x_2d.dtype, device=x_2d.device)
    img[0, :C, :] = x_2d

    return img



# ─────────────────────────────────────────────
# 1. 일반 다변량 CSV Dataset (Electricity, ETT, Weather 등)
# ─────────────────────────────────────────────
class CSV2DDataset(Dataset):
    def __init__(
        self,
        csv_path,
        seq_len=512,
        patch_size=8,
        stride=1,
        n_views=4,
        noise_std=0.02,
        scale_range=(0.9, 1.1),
        mode="train",
    ):
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.n_views = n_views if mode == "train" else 1
        self.noise_std = noise_std if mode == "train" else 0.0
        self.scale_range = scale_range if mode == "train" else (1.0, 1.0)
        self.mode = mode

        # 데이터 로드
        df_np = self._load_and_preprocess(csv_path)
        total_len = len(df_np)

        # Train: 70%, Val: 10%, Test: 20%
        val_len = int(total_len * 0.1)
        test_len = int(total_len * 0.2)
        train_len = total_len - val_len - test_len

        # 정규화 (Train 통계로만 fit)
        scaler = StandardScaler()
        scaler.fit(df_np[:train_len])
        df_norm = scaler.transform(df_np)

        if mode == "train":
            split_data = df_norm[:train_len]
        elif mode == "val":
            split_data = df_norm[train_len : train_len + val_len]
        elif mode == "test":
            split_data = df_norm[train_len + val_len :]
        else:
            raise ValueError("mode must be train, val, or test")

        self.data = torch.from_numpy(split_data).float()  # [Total_T, C]
        self.C = self.data.shape[1]

        # ViT 입력 이미지 H, W 지정용
        self.H = math.ceil(self.C / self.patch_size) * self.patch_size
        self.W = self.seq_len

        # 슬라이딩 인덱스 미리 계산 (속도 최적화)
        self.indices = []
        data_len = len(self.data)
        for i in range(0, data_len - self.seq_len + 1, self.stride):
            self.indices.append(i)

        print(f"✅ CSV2DDataset ({mode}): {len(self.indices)} 샘플 | 변수(C)={self.C} → H={self.H}, W={self.W}")

    def _load_and_preprocess(self, path):
        df = pd.read_csv(path, low_memory=False)
        timestamp_col = None
        for c in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
            if c in df.columns:
                timestamp_col = c
                break
        
        if timestamp_col:
            df.sort_values(by=[timestamp_col], inplace=True)
            df.drop(columns=[timestamp_col], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df.loc[df[col] < -9990.0, col] = np.nan
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df.dropna(inplace=True, axis=0)

        fcols = df.select_dtypes("float").columns
        df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
        icols = df.select_dtypes("integer").columns
        df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")

        return df.values.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start : start + self.seq_len]  # [T, C]
        window = window.T  # [C, T] 로 변환

        views = []
        for _ in range(self.n_views):
            if self.mode == "train":
                w_aug = augment_timeseries(window, self.noise_std, self.scale_range)
            else:
                w_aug = window.clone()
                
            img = ts_to_1ch_image(w_aug, self.patch_size)  # [1, H, W]
            views.append(img)
            
        views = torch.stack(views)  # [V, 1, H, W]
        label = torch.tensor(0, dtype=torch.long)
        return views, label


# ─────────────────────────────────────────────
# 2. TSLD 대규모 데이터셋 (Lazy Loading)
# ─────────────────────────────────────────────
class TSLD2DDataset(Dataset):
    def __init__(
        self,
        root_path,
        seq_len=512,
        patch_size=8,
        stride=512,  # TSLD는 기본적으로 데이터가 많아 non-overlapping 권장
        n_views=4,
        noise_std=0.02,
        scale_range=(0.9, 1.1),
        mode="train",
        max_files=None,
    ):
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.n_views = n_views if mode == "train" else 1
        self.noise_std = noise_std if mode == "train" else 0.0
        self.scale_range = scale_range if mode == "train" else (1.0, 1.0)
        self.mode = mode

        # 단일 변수(C=1)이므로 H=patch_size
        self.C = 1
        self.H = math.ceil(self.C / self.patch_size) * self.patch_size  # 8
        self.W = self.seq_len

        csv_files = []
        for root, dirs, files in os.walk(root_path):
            for f in files:
                if f.endswith(".csv") and "hhh" not in f:
                    csv_files.append(os.path.join(root, f))
        
        if max_files:
            csv_files = csv_files[:max_files]

        print(f"📊 TSLD 2D Dataset ({mode}): {len(csv_files)}개 파일 로드 준비")

        self.time_series_list = []  # List[[T, 1]]
        self.sample_indices = []

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    df.loc[df[col] < -9990.0, col] = np.nan
                df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
                df.dropna(inplace=True, axis=0)

                timestamp_col = None
                for c in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
                    if c in df.columns:
                        timestamp_col = c
                        break
                
                feature_cols = [c for c in df.columns if c != timestamp_col] if timestamp_col else df.columns.tolist()

                for col in feature_cols:
                    col_data = df[col].values.astype(np.float32)
                    col_data = pd.Series(col_data).ffill().fillna(0).values
                    col_data = torch.tensor(col_data).float().unsqueeze(1) # [T, 1]

                    total_len = len(col_data)
                    val_len = int(total_len * 0.1)
                    test_len = int(total_len * 0.2)
                    train_len = total_len - val_len - test_len

                    # Train 통계로 z-score
                    train_mean = col_data[:train_len].mean(0)
                    train_std = col_data[:train_len].std(0)
                    col_data = (col_data - train_mean) / (train_std + 1e-8)

                    if mode == "train":
                        split_data = col_data[:train_len]
                    elif mode == "val":
                        split_data = col_data[train_len : train_len + val_len]
                    elif mode == "test":
                        split_data = col_data[train_len + val_len :]
                    else:
                        raise ValueError("mode must be train, val, or test")

                    if len(split_data) < self.seq_len:
                        continue

                    series_idx = len(self.time_series_list)
                    self.time_series_list.append(split_data)

                    for start_idx in range(0, len(split_data) - self.seq_len + 1, self.stride):
                        self.sample_indices.append((series_idx, start_idx))
            except Exception as e:
                pass
        
        print(f"✅ TSLD2DDataset ({mode}): {len(self.sample_indices)} 샘플 | H={self.H}, W={self.W}")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        series_idx, start_idx = self.sample_indices[idx]
        window = self.time_series_list[series_idx][start_idx : start_idx + self.seq_len]  # [T, 1]
        window = window.T  # [1, T]

        views = []
        for _ in range(self.n_views):
            if self.mode == "train":
                w_aug = augment_timeseries(window, self.noise_std, self.scale_range)
            else:
                w_aug = window.clone()
                
            img = ts_to_1ch_image(w_aug, self.patch_size)  # [1, H, W]
            views.append(img)
            
        views = torch.stack(views)  # [V, 1, H, W]
        label = torch.tensor(0, dtype=torch.long)
        return views, label


def get_2d_loaders(
    dataset_type="electricity",  # "electricity" or "tsld"
    path="/data/pjh_workspace/Dataset/long_term_forecast/electricity/electricity.csv",
    batch_size=64,
    seq_len=512,
    patch_size=8,
    stride=None,
    n_views=4,
    num_workers=4,
    max_files=None,  # TSLD 용
):
    """
    Train / Val 2D DataLoader 리턴.
    """
    if dataset_type == "tsld":
        if stride is None:
            stride = seq_len  # TSLD 기본은 Non-overlapping
            
        train_ds = TSLD2DDataset(path, seq_len=seq_len, patch_size=patch_size, stride=stride, n_views=n_views, mode="train", max_files=max_files)
        val_ds = TSLD2DDataset(path, seq_len=seq_len, patch_size=patch_size, stride=stride, n_views=1, mode="val", max_files=max_files)
    else: # electricity, ett 등
        if stride is None:
            stride = 64  # 기본 stride
            
        train_ds = CSV2DDataset(path, seq_len=seq_len, patch_size=patch_size, stride=stride, n_views=n_views, mode="train")
        val_ds = CSV2DDataset(path, seq_len=seq_len, patch_size=patch_size, stride=stride, n_views=1, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

