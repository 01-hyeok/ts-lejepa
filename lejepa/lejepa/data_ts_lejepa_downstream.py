import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class CSVDownstreamDataset(Dataset):
    """
    Linear Probing / Forecasting용 데이터셋 (seq_len -> pred_len 예측)
    TS-JEPA 기준 분할(7:1:2 or 6:2:2 등 데이터셋 특성 반영) 및 정규화(StandardScaler)
    """
    def __init__(self, data_path, dataset_type="ETTm1", seq_len=512, pred_len=96, mode="train"):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.dataset_type = dataset_type
        
        # 데이터 분할 비율 설정
        if dataset_type == "ETTm1" or dataset_type == "ETTm2":
            # ETT 계열: 총 12 + 4 + 4 months (비율 6:2:2와 거의 유사)
            train_ratio, val_ratio, test_ratio = 12/20, 4/20, 4/20
        else: # weather 등
            train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2

        # 1. 파일 읽기 및 전처리 (날짜 컬럼 제거, 이상치 제거 등)
        df_np = self._load_and_preprocess(data_path)
        total_len = len(df_np)
        
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        # Test는 나머지 (계산 오차 방지)
        
        # 2. 정규화 (Train 기준으로 핏팅)
        self.scaler = StandardScaler()
        self.scaler.fit(df_np[:train_len])
        df_norm = self.scaler.transform(df_np)
        
        # 3. Mode에 따른 데이터프레임 할당
        if mode == "train":
            # TS-JEPA처럼 Train은 Train구간 안에서 시작~끝을 잡음
            start_idx = 0
            end_idx = train_len
        elif mode == "val":
            # Validation은 이전 seq_len 만큼 겹쳐서 시작 (정답 예측용 context 확보)
            start_idx = train_len - self.seq_len
            end_idx = train_len + val_len
        else: # test
            start_idx = train_len + val_len - self.seq_len
            end_idx = total_len
            
        self.data = torch.from_numpy(df_norm[start_idx:end_idx]).float()
        
        # 4. 샘플 구성
        # context(seq_len) + pred_len 이 가능하도록 인덱스 구성
        self.indices = [i for i in range(0, len(self.data) - self.seq_len - self.pred_len + 1)]
        print(f"✅ CSVDownstreamDataset ({mode}): {len(self.indices)} 샘플 로드 완료 ({dataset_type})")

    def _load_and_preprocess(self, path):
        df = pd.read_csv(path, low_memory=False)
        for c in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
            if c in df.columns: 
                df.drop(columns=[c], inplace=True)
                break
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 극단 이상치 처리 (TS-JEPA 방식)
        for col in numeric_cols:
            df.loc[df[col] < -9990.0, col] = np.nan
            
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        df.dropna(inplace=True, axis=0) # 그래도 남은 nan 제거
        return df.values.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        context = self.data[start : start + self.seq_len].T # [C, Seq_len]
        target = self.data[start + self.seq_len : start + self.seq_len + self.pred_len].T # [C, Pred_len]
        
        return context, target

def get_downstream_loaders(data_path, dataset_type, batch_size=32, seq_len=512, pred_len=96, num_workers=4):
    train_ds = CSVDownstreamDataset(data_path, dataset_type, seq_len, pred_len, mode="train")
    val_ds = CSVDownstreamDataset(data_path, dataset_type, seq_len, pred_len, mode="val")
    test_ds = CSVDownstreamDataset(data_path, dataset_type, seq_len, pred_len, mode="test")
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, test_ds.scaler
