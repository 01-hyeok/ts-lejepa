"""
TS-JEPA 데이터셋 모듈
=====================
기존 MultiResolution 변환을 제거하고, 단순 [C, T] 윈도우를 반환합니다.
마스킹은 학습 스크립트(run_ts_jepa.py)에서 배치 단위로 처리됩니다.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class CSVJEPADataset(Dataset):
    """단일 CSV 파일 기반 TS-JEPA 데이터셋.

    JEPA는 마스킹을 모델 내부에서 처리하므로,
    데이터셋은 원시 [C, T] 윈도우만 반환합니다.

    Args:
        csv_path: CSV 파일 경로.
        seq_len: 윈도우 길이 (시간 단위).
        stride: 슬라이딩 윈도우 간격.
        mode: "train" | "val" | "test".
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int = 512,
        stride: int = 128,
        mode: str = "train",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.mode = mode

        # 전처리 및 분할
        df_np = self._load_and_preprocess(csv_path)
        total_len = len(df_np)

        val_len = int(total_len * 0.1)
        test_len = int(total_len * 0.2)
        train_len = total_len - val_len - test_len

        # StandardScaler: train 통계량으로 전체 정규화
        scaler = StandardScaler()
        scaler.fit(df_np[:train_len])
        df_norm = scaler.transform(df_np)

        if mode == "train":
            split_data = df_norm[:train_len]
        elif mode == "val":
            split_data = df_norm[train_len : train_len + val_len]
        else:
            split_data = df_norm[train_len + val_len :]

        self.data = torch.from_numpy(split_data.astype(np.float32))  # [T_split, C]
        self.indices = list(range(0, len(self.data) - seq_len + 1, stride))
        print(f"✅ CSVJEPADataset ({mode}): {len(self.indices)} 샘플 로드 완료 | C={self.data.shape[1]}, T={seq_len}")

    def _load_and_preprocess(self, path: str) -> np.ndarray:
        df = pd.read_csv(path, low_memory=False)
        # 날짜/타임스탬프 컬럼 제거
        for c in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
                break
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # 극단 이상치 처리 (-9990 이하: Weather 데이터셋)
        for col in numeric_cols:
            df.loc[df[col] < -9990.0, col] = np.nan
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df.dropna(inplace=True, axis=0)
        return df.values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            [C, T] 원시 시계열 윈도우.
        """
        start = self.indices[idx]
        window = self.data[start : start + self.seq_len]  # [T, C]
        return window.T  # [C, T]


class TSLDJEPADataset(Dataset):
    """TSLD (Time Series Large Dataset) 기반 TS-JEPA 데이터셋.

    단변량 시계열 파일들을 Lazy Loading으로 인덱싱합니다.
    각 샘플은 [1, T] 형태로 반환됩니다.

    Args:
        root_path: TSLD 루트 디렉토리.
        seq_len: 윈도우 길이.
        stride: 슬라이딩 윈도우 간격.
        mode: "train" | "val" | "test".
        max_files: 최대 파일 수 (None이면 전체).
    """

    def __init__(
        self,
        root_path: str,
        seq_len: int = 512,
        stride: int = 512,
        mode: str = "train",
        max_files: int = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride
        self.mode = mode

        self.sample_indices: list[tuple[int, int]] = []  # (series_idx, start)
        self.time_series_list: list[np.ndarray] = []

        csv_files = sorted(
            [
                os.path.join(r, f)
                for r, _, fs in os.walk(root_path)
                for f in fs
                if f.endswith(".csv") and "hhh" not in f
            ]
        )
        if max_files:
            csv_files = csv_files[:max_files]

        print(f"🔍 TSLDJEPADataset ({mode}) 인덱스 생성 중...")
        for file_path in csv_files:
            try:
                df_meta = pd.read_csv(file_path, nrows=5)
                cols = [
                    c
                    for c in df_meta.columns
                    if c not in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]
                ]
                df = pd.read_csv(file_path, usecols=cols)
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df = df.ffill().bfill().fillna(0)

                for col in cols:
                    col_data = df[col].values.astype(np.float32)
                    col_data = pd.Series(col_data).ffill().fillna(0).values
                    total_len = len(col_data)
                    if total_len < self.seq_len:
                        continue

                    val_len = int(total_len * 0.1)
                    test_len = int(total_len * 0.2)
                    train_len = total_len - val_len - test_len
                    if train_len <= 0:
                        continue

                    m = col_data[:train_len].mean()
                    s = col_data[:train_len].std() + 1e-8

                    if mode == "train":
                        start_limit, offset = train_len, 0
                    elif mode == "val":
                        start_limit, offset = train_len + val_len, train_len
                    else:
                        start_limit, offset = total_len, train_len + val_len

                    current_len = start_limit - offset
                    if current_len >= self.seq_len:
                        series_idx = len(self.time_series_list)
                        split_data = ((col_data[offset:start_limit] - m) / s).reshape(-1, 1).astype(np.float32)
                        self.time_series_list.append(split_data)
                        for start in range(0, current_len - self.seq_len + 1, self.stride):
                            self.sample_indices.append((series_idx, start))
            except Exception:
                continue

        print(f"✅ TSLDJEPADataset ({mode}): {len(self.sample_indices)} 샘플 인덱싱 완료")

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            [1, T] — 단변량 시계열 윈도우 (채널 수 = 1).
        """
        series_idx, start = self.sample_indices[idx]
        series_data = self.time_series_list[series_idx]  # [T_split, 1]
        window = series_data[start : start + self.seq_len]  # [T, 1]
        return torch.from_numpy(window.T)  # [1, T]


def get_jepa_loaders(
    dataset_type: str = "electricity",
    path: str = "",
    batch_size: int = 64,
    seq_len: int = 512,
    stride: int = 128,
    num_workers: int = 4,
    max_files: int = None,
) -> tuple[DataLoader, DataLoader]:
    """TS-JEPA 전용 DataLoader 생성.

    Args:
        dataset_type: "tsld" 또는 CSV 파일 경로 기반 타입.
        path: 데이터 경로.
        batch_size: 배치 크기.
        seq_len: 시퀀스 길이.
        stride: 슬라이딩 스트라이드.
        num_workers: DataLoader 워커 수.
        max_files: TSLD에서 최대 파일 수.

    Returns:
        (train_loader, val_loader) 튜플.
    """
    if dataset_type == "tsld":
        train_ds = TSLDJEPADataset(path, seq_len, seq_len, "train", max_files)
        val_ds = TSLDJEPADataset(path, seq_len, seq_len, "val", max_files)
    else:
        train_ds = CSVJEPADataset(path, seq_len, stride, "train")
        val_ds = CSVJEPADataset(path, seq_len, stride, "val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
