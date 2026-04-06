"""
data_ts_classification.py
Classification dataset for LeJEPA-based time-series classification.

Supports two formats:
  1. UCR/UEA .ts format  (via sktime-style parser, labels in first column)
  2. Generic CSV         (last column = integer label, all others = time-series features)

Consistent with data_ts_lejepa_downstream.py conventions:
  - StandardScaler fitted on train split
  - Returns (x: [C, T], label: int) pairs
"""
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ts_file(path: str):
    """
    Minimal parser for UCR/UEA .ts format.
    Returns (X: np.ndarray [N, C, T], y: np.ndarray [N] with str labels).
    Assumes all series have equal length and univariate or multivariate.
    """
    data_started = False
    series_list, labels = [], []
    n_dims = 1

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("@data"):
                data_started = True
                continue
            if line.lower().startswith("@classlabel"):
                continue
            if line.lower().startswith("@"):
                if "numdimensions" in line.lower():
                    n_dims = int(re.search(r"\d+", line).group())
                continue
            if data_started:
                # format: dim1:val,val,... dim2:val,val,... label
                # OR: val,val,...,label  (univariate flat)
                if ":" in line:
                    # multivariate colon-separated
                    *dims_raw, label = line.rsplit(":", 1)
                    # dims_raw is a single joined string; rebuild
                    parts = (":".join(dims_raw)).split(":")
                    # last token may be label if no trailing colon
                    if len(parts) == n_dims + 1:
                        label = parts[-1].strip()
                        parts = parts[:-1]
                    elif len(parts) == n_dims:
                        label = label.strip()

                    # Handle '?' as NaN explicitly for UEA datasets
                    dim_arrays = []
                    for p in parts:
                        p_clean = p.replace('?', 'NaN')
                        arr = np.fromstring(p_clean.strip(), sep=",")
                        # Fill NaNs (sktime/TSLib convention)
                        if np.isnan(arr).any():
                            arr = pd.Series(arr).interpolate(limit_direction='both').fillna(0).values
                        dim_arrays.append(arr)
                    series_list.append(np.stack(dim_arrays, axis=0))  # [C, T_i]
                else:
                    # univariate, comma-separated, last value = label
                    vals = line.replace('?', 'NaN').split(",")
                    label = vals[-1].strip()
                    arr = np.array([float(v) for v in vals[:-1]])
                    if np.isnan(arr).any():
                        arr = pd.Series(arr).interpolate(limit_direction='both').fillna(0).values
                    series_list.append(arr[np.newaxis, :])             # [1, T_i]
                labels.append(label)

    # Handle unequal lengths (Very common in some UEA datasets like JapaneseVowels)
    max_len = max(s.shape[1] for s in series_list)
    padded_series = []
    for s in series_list:
        if s.shape[1] < max_len:
            pad_w = max_len - s.shape[1]
            # Zero pad on the right (typical for UEA unequal lengths)
            s = np.pad(s, ((0, 0), (0, pad_w)))
        padded_series.append(s)

    X = np.stack(padded_series, axis=0).astype(np.float32)  # [N, C, max_T]
    y = np.array(labels)
    return X, y


def _load_csv_classification(path: str):
    """
    CSV where ALL columns except the last are feature columns,
    and the last column contains integer/string labels.
    Returns (X: [N, T, C], y: [N] str/int labels).
    """
    df = pd.read_csv(path, low_memory=False)
    label_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    y = df[label_col].values.astype(str)
    X = df[feature_cols].values.astype(np.float32)          # [N, features]
    # Treat each row as a univariate series of length=features, C=1
    X = X[:, np.newaxis, :]                                  # [N, 1, T]
    return X, y


def _parse_ucr_txt_file(path: str):
    """
    UCR .txt / .tsv format parser.
    Each row: <label>  <v1>  <v2>  ...  (tab- or space-separated, first col = label)
    Returns (X: np.ndarray [N, 1, T], y: np.ndarray [N] with str labels).
    """
    series_list, labels = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            label = tokens[0]
            vals  = np.array([float(v) for v in tokens[1:]], dtype=np.float32)
            series_list.append(vals)
            labels.append(label)
    # handle unequal lengths (pad right with zeros)
    max_len = max(len(s) for s in series_list)
    padded = np.zeros((len(series_list), max_len), dtype=np.float32)
    for i, s in enumerate(series_list):
        padded[i, :len(s)] = s
    X = padded[:, np.newaxis, :]   # [N, 1, T]
    y = np.array(labels)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TSClassificationDataset(Dataset):
    """
    Unified classification dataset.

    Args:
        data_root: path to a directory containing TRAIN.ts / TEST.ts,
                   or a CSV file with the last column as label.
        seq_len:   fixed sequence length (pad/truncate to this length).
        mode:      'train' | 'val' | 'test'.
        scaler:    fitted StandardScaler to apply (None = fit from data).
        le:        fitted LabelEncoder (None = fit from data).
        val_ratio: fraction of training data used for validation split.
    """

    def __init__(
        self,
        data_root: str,
        seq_len: int = 512,
        mode: str = "train",
        scaler: StandardScaler | None = None,
        le: LabelEncoder | None = None,
        val_ratio: float = 0.1,
    ):
        self.seq_len = seq_len
        self.mode = mode

        X_raw, y_raw = self._load(data_root, mode, val_ratio)
        # X_raw: [N, C, T_orig]

        # ── normalise per-channel (fit on train only) ──────────────────────
        N, C, T = X_raw.shape
        X_flat = X_raw.transpose(0, 2, 1).reshape(-1, C)   # [N*T, C]
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(X_flat)
        X_flat_norm = scaler.transform(X_flat)
        X_norm = X_flat_norm.reshape(N, T, C).transpose(0, 2, 1)  # [N, C, T]
        self.scaler = scaler

        # ── encode labels ─────────────────────────────────────────────────
        if le is None:
            le = LabelEncoder()
            # TSLib Behavior: Fit LabelEncoder on ALL possible labels in the dataset
            # to prevent 'unseen label' errors if a class is missing in the train split.
            if os.path.isdir(data_root):
                try:
                    p1 = _find_file(data_root, ["TRAIN.ts", "TRAIN.txt", f"{os.path.basename(data_root)}_TRAIN.ts"])
                    p2 = _find_file(data_root, ["TEST.ts",  "TEST.txt", f"{os.path.basename(data_root)}_TEST.ts"])
                    _, l1 = _parse_ts_file(p1)
                    _, l2 = _parse_ts_file(p2)
                    le.fit(np.concatenate((l1, l2)))
                except Exception as e:
                    le.fit(y_raw)
            else:
                _, l_all = _load_csv_classification(data_root)
                le.fit(l_all)
        self.le = le
        self.num_classes = len(le.classes_)

        # ── pad / truncate time axis ──────────────────────────────────────
        T_orig = X_norm.shape[-1]
        
        # Auto-detect sequence length if <= 0
        if seq_len <= 0:
            seq_len = T_orig
            
        if T_orig >= seq_len:
            X_norm = X_norm[:, :, -seq_len:]           # keep last seq_len
        else:
            pad_w = seq_len - T_orig
            X_norm = np.pad(X_norm, ((0, 0), (0, 0), (pad_w, 0)))  # left-pad

        self.X = torch.from_numpy(X_norm).float()                   # [N, C, T]
        self.y = torch.from_numpy(le.transform(y_raw)).long()       # [N]

        print(f"✅ TSClassificationDataset ({mode}): {len(self.X)} samples | "
              f"C={C} T={seq_len} classes={self.num_classes}")

    # ── internal loader ────────────────────────────────────────────────────

    def _load(self, data_root: str, mode: str, val_ratio: float):
        """Dispatch to .ts or .csv loader.  Returns (X [N,C,T], y [N])."""
        
        # ── Handle ZIP files automatically (TSLib behavior) ──
        if not os.path.exists(data_root) and os.path.exists(data_root + ".zip"):
            import zipfile
            print(f"📦 Auto-unzipping {data_root}.zip ...")
            with zipfile.ZipFile(data_root + ".zip", 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(data_root))

        # Case 1: directory with TRAIN / TEST files (.ts or .txt)
        if os.path.isdir(data_root):
            dname = os.path.basename(data_root)
            train_path = _find_file(data_root, [
                f"{dname}_TRAIN.ts", "TRAIN.ts",
                f"{dname}_TRAIN.txt", "TRAIN.txt",
                f"{dname}_TRAIN.tsv", "TRAIN.tsv",
            ])
            test_path = _find_file(data_root, [
                f"{dname}_TEST.ts", "TEST.ts",
                f"{dname}_TEST.txt", "TEST.txt",
                f"{dname}_TEST.tsv", "TEST.tsv",
            ])

            # 파일 확장자에 따라 파서 선택
            def _parse(p):
                ext = os.path.splitext(p)[1].lower()
                if ext in (".txt", ".tsv"):
                    return _parse_ucr_txt_file(p)
                return _parse_ts_file(p)

            X_train, y_train = _parse(train_path)
            X_test,  y_test  = _parse(test_path)

            if mode == "test":
                return X_test, y_test

            # split train -> train / val
            N = len(X_train)
            n_val = max(1, int(N * val_ratio))
            n_train = N - n_val

            # 데이터 셔플링 (Train과 Val 분할 전)
            rng = np.random.default_rng(42)  # 분할이 양쪽에서 동일하게 일어나도록 시드 고정
            indices = rng.permutation(N)
            X_train = X_train[indices]
            y_train = y_train[indices]

            if mode == "train":
                return X_train[:n_train], y_train[:n_train]
            else:  # val
                return X_train[n_train:], y_train[n_train:]

        # Case 2: single CSV file
        if os.path.isfile(data_root) and data_root.endswith(".csv"):
            X_all, y_all = _load_csv_classification(data_root)
            N = len(X_all)
            n_test  = max(1, int(N * 0.2))
            n_val   = max(1, int(N * val_ratio))
            n_train = N - n_val - n_test

            # 데이터 셔플링 (CSV 통짜 데이터를 분할 전)
            rng = np.random.default_rng(42)
            indices = rng.permutation(N)
            X_all = X_all[indices]
            y_all = y_all[indices]

            if mode == "train":
                return X_all[:n_train], y_all[:n_train]
            elif mode == "val":
                return X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
            else:
                return X_all[n_train + n_val:], y_all[n_train + n_val:]

        raise ValueError(f"data_root must be a directory with .ts files or a CSV file. Got: {data_root}")

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   # ([C, T], scalar long)


def _find_file(directory: str, candidates: list[str]) -> str:
    for name in candidates:
        p = os.path.join(directory, name)
        if os.path.exists(p):
            return p
    # fallback: first .ts or .txt file found in directory
    for ext in (".ts", ".txt", ".tsv"):
        for fname in os.listdir(directory):
            if fname.endswith(ext):
                return os.path.join(directory, fname)
    raise FileNotFoundError(f"No .ts/.txt/.tsv file found in {directory}. Tried: {candidates}")


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory  — mirrors get_downstream_loaders() convention
# ─────────────────────────────────────────────────────────────────────────────

def get_classification_loaders(
    data_root: str,
    seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    val_ratio: float = 0.1,
):
    """
    Returns (train_loader, val_loader, test_loader, num_classes, in_vars).
    Scaler and LabelEncoder are fitted on train split and shared across splits.
    """
    train_ds = TSClassificationDataset(data_root, seq_len, mode="train",  val_ratio=val_ratio)
    
    # If seq_len was auto (<= 0), lock the discovered length for val and test
    actual_seq_len = train_ds.X.shape[-1]
    
    val_ds   = TSClassificationDataset(data_root, actual_seq_len, mode="val",
                                       scaler=train_ds.scaler, le=train_ds.le, val_ratio=val_ratio)
    test_ds  = TSClassificationDataset(data_root, actual_seq_len, mode="test",
                                       scaler=train_ds.scaler, le=train_ds.le, val_ratio=val_ratio)

    # drop_last only when there are enough samples to form at least one full batch
    # (prevents empty DataLoader on small UEA datasets like PEMS-SF)
    do_drop_last = len(train_ds) > batch_size

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  drop_last=do_drop_last, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    in_vars = train_ds.X.shape[1]  # C
    return train_loader, val_loader, test_loader, train_ds.num_classes, in_vars, actual_seq_len
