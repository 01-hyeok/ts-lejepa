
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add repo root to path
repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from lejepa.data_ts_lejepa_basic import TSLDMultiResDataset

def verify():
    root_path = '/data/pjh_workspace/Dataset/TSLD-1G'
    seq_len = 512
    
    print(f"--- LeJEPA TSLD Verification (stride={seq_len}) ---")
    for mode in ['train', 'val', 'test']:
        ds = TSLDMultiResDataset(
            root_path=root_path,
            seq_len=seq_len,
            mode=mode,
            max_files=10
        )
        print(f"[{mode.upper()}] Samples: {len(ds)}")
        if len(ds) > 0:
            x = ds[0]
            # LeJEPA returns [1, T] tensor
            print(f"  Shape: {x.shape} (Mean: {x.mean():.4f}, Std: {x.std():.4f})")

if __name__ == "__main__":
    verify()
