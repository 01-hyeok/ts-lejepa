import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

from lejepa.data_ts_lejepa_downstream import get_downstream_loaders
from lejepa.model_ts_lejepa_basic import MultiResViTEncoder
from lejepa.model_ts_lejepa_tiling import MultiResViTTilingEncoder
from lejepa.model_ts_tivit import TiViTIndependentEncoder, TiViTDependentEncoder
from lejepa.model_ts_timevlm import TimeVLMEncoder
from lejepa.model_ts_conv2d import Conv2DLearnableEncoder
from lejepa.revin import RevIN
from lejepa.model_ts_lejepa_ci import MultiResViTCIEncoder
from lejepa.model_ts_lejepa_1d import PatchTS1DEncoder
from lejepa.model_ts_utica import UTICAEncoder
from lejepa.model_ts_conv import MultiResViTConvEncoder

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

class LeJEPALinearProbingHead(nn.Module):
    def __init__(self, in_vars, embed_dim, pred_len):
        super().__init__()
        # LeJEPA 백본은 C차원을 Projection해서 [B, Embed_Dim] 1개 백터를 내뱉습니다.
        # 타겟은 [B, C, Pred_Len]이므로 Linear 계층 1개로 한 번에 복원합니다.
        self.head = nn.Linear(embed_dim, in_vars * pred_len)
        self.in_vars = in_vars
        self.pred_len = pred_len
        
    def forward(self, x):
        # x: [B, Embed]
        # output: [B, C, Pred_Len]
        out = self.head(x)
        return out.view(-1, self.in_vars, self.pred_len)

class LeJEPALinearProbingHeadCI(nn.Module):
    def __init__(self, in_vars, embed_dim, pred_len):
        super().__init__()
        # 1채널 임베딩을 받아 1채널 길이로 출력
        self.head = nn.Linear(embed_dim, pred_len)
        self.in_vars = in_vars
        self.pred_len = pred_len
        
    def forward(self, x):
        # x: [B * C, Embed]
        out = self.head(x)
        # output: [B, C, Pred_Len]
        return out.view(-1, self.in_vars, self.pred_len)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. DataLoader
    train_loader, val_loader, test_loader, scaler = get_downstream_loaders(
        data_path=args.data_path,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_workers=args.num_workers
    )
    
    first_batch_c, _ = next(iter(train_loader))
    in_vars = first_batch_c.shape[1] # Target Data Channels (예: 7)
    
    arch_key = args.arch.lower().replace('-', '_')
    
    # 2. Pretrained Dataset 차원(pretrain_in_vars) 동적 확인
    pretrain_in_vars = in_vars
    state_dict = None
    if os.path.exists(args.pretrain_path):
        ckpt = torch.load(args.pretrain_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # 체크포인트의 가중치 Shape으로부터 pretrain_in_vars 추론
        if arch_key == "basic" and "proj.weight" in state_dict:
            pretrain_in_vars = state_dict["proj.weight"].shape[1]
        elif arch_key in ["tiling", "tiling_repeat", "tivit_dep"] and "backbone.patch_embed.proj.weight" in state_dict:
            pretrain_in_vars = state_dict["backbone.patch_embed.proj.weight"].shape[1]
        elif arch_key == "conv2d" and "canvas_encoder.proj.weight" in state_dict:
            pretrain_in_vars = state_dict["canvas_encoder.proj.weight"].shape[1]
        elif arch_key in ["timevlm", "tiling_ci", "patchtst", "utica", "conv"]:
            pretrain_in_vars = in_vars # channel-agnostic or 1D architecture
        else:
            pretrain_in_vars = 321 if args.pretrain_dataset == "electricity" else in_vars
    else:
        pretrain_in_vars = 321 if args.pretrain_dataset == "electricity" else in_vars
    
    print("="*50)
    print(f"🚀 Architecture: {args.arch.upper()}")
    print(f"📊 Target Dataset: {args.dataset_type} (Channels: {in_vars})")
    print(f"📊 Pretrain Info: {args.pretrain_dataset} (Detected Channels: {pretrain_in_vars})")
    print(f"📊 Prediction Length: {args.pred_len}")
    print("="*50)
    
    # 3. Pretrained Encoder 로드 & Freeze
    if arch_key == "basic":
        encoder = MultiResViTEncoder(in_vars=pretrain_in_vars, model_name=args.vit_model, proj_dim=args.proj_dim).to(device)
    elif arch_key in ["tiling", "tiling_repeat"]:
        encoder = MultiResViTTilingEncoder(in_vars=pretrain_in_vars, model_name=args.vit_model, proj_dim=args.proj_dim).to(device)
    elif arch_key == "tivit_indep":
        encoder = TiViTIndependentEncoder(in_vars=pretrain_in_vars, model_name=args.vit_model, proj_dim=args.proj_dim).to(device)
    elif arch_key == "tivit_dep":
        encoder = TiViTDependentEncoder(in_vars=pretrain_in_vars, model_name=args.vit_model, proj_dim=args.proj_dim).to(device)
    elif arch_key == "timevlm":
        # TimeVLM은 pretrain_in_vars를 받아 내부 이미지를 생성하고 ViT(in_chans=3)로 전달함
        encoder = TimeVLMEncoder(in_vars=pretrain_in_vars, model_name=args.vit_model, proj_dim=args.proj_dim).to(device)
    elif arch_key == "conv2d":
        # Conv2D: [B,1,C,T]를 2D 이미지로 보고 학습 가능한 Conv2D 체인으로 캔버스 생성
        # Conv2D 가중치는 in_vars에 무관하므로 pretrain_in_vars로 초기화 (타기에서 동일하게 동작)
        encoder = Conv2DLearnableEncoder(in_vars=pretrain_in_vars, model_name=args.vit_model, proj_dim=args.proj_dim).to(device)
    elif arch_key == "tiling_ci":
        encoder = MultiResViTCIEncoder(in_vars=pretrain_in_vars, model_name=args.vit_model, proj_dim=args.proj_dim).to(device)
    elif arch_key == "patchtst":
        encoder = PatchTS1DEncoder(in_vars=pretrain_in_vars, d_model=384, proj_dim=args.proj_dim, use_revin=args.use_revin).to(device)
    elif arch_key == "utica":
        encoder = UTICAEncoder(in_vars=pretrain_in_vars, d_model=384, proj_dim=args.proj_dim, use_revin=args.use_revin).to(device)
    elif arch_key == "conv":
        encoder = MultiResViTConvEncoder(in_vars=pretrain_in_vars, model_name=args.vit_model, proj_dim=args.proj_dim).to(device)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    if state_dict is not None:
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        matched_keys = len(state_dict) - len(missing)
        print(f"✅ Loaded pretrained encoder from {args.pretrain_path}")
        print(f"   - {matched_keys}/{len(state_dict)} keys matched.")
        if missing:
            print(f"   ⚠️ Missing keys (not loaded): {missing[:3]}...")
    else:
        print(f"⚠️ Pretrained path not found: {args.pretrain_path}")
        
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    # 3. Channel Adapter 디자인
    if pretrain_in_vars != in_vars:
        channel_adapter = nn.Linear(in_vars, pretrain_in_vars).to(device)
        if arch_key == "basic":
            # Basic: 채널들을 자유롭게 섞어서(Mixing) 사전 학습 차원에 맞춤
            print(f"✅ Adapter [BASIC]: Linear Mixing {in_vars} -> {pretrain_in_vars}")
        else:
            # Tiling/TiViT/TimeVLM: 채널 고유의 이미지가 있으므로 섞지 않고 보존하되,
            # pretrain_in_vars개 채널을 모두 활용하기 위해 반복(Repeat) 방식으로 초기화
            with torch.no_grad():
                channel_adapter.weight.zero_()
                for j in range(pretrain_in_vars):
                    channel_adapter.weight[j, j % in_vars] = 1.0
            print(f"✅ Adapter [{arch_key.upper()}]: Repeat-based Linear {in_vars} -> {pretrain_in_vars}")
    else:
        channel_adapter = nn.Identity().to(device)

    # 4. Probing Head & RevIN
    if arch_key in ["patchtst", "utica"]:
        embed_dim = encoder.encoder.d_model if arch_key == "utica" else encoder.d_model
    else:
        embed_dim = encoder.backbone.num_features
        
    if arch_key in ["tiling_ci", "patchtst", "utica"]:
        decoder = LeJEPALinearProbingHeadCI(in_vars, embed_dim, args.pred_len).to(device)
        print(f"✅ Probing Head [CI]: [B*C, {embed_dim}] -> [B, {in_vars}, {args.pred_len}]")
    else:
        decoder = LeJEPALinearProbingHead(in_vars, embed_dim, args.pred_len).to(device)
        print(f"✅ Probing Head: [B, {embed_dim}] -> [B, {in_vars}, {args.pred_len}]")
    
    revin = None
    if args.use_revin:
        revin = RevIN(num_features=in_vars, affine=False).to(device)
    
    # 5. Optimizer (Adapter + Decoder만 학습)
    trainable_params = list(decoder.parameters())
    if pretrain_in_vars != in_vars:
        trainable_params += list(channel_adapter.parameters())
        
    opt = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_dir, f'linear_{args.arch}_{args.dataset_type}_{args.pred_len}'))

    best_val_loss = float('inf')

    # 6. Training Loop
    import time
    for epoch in range(args.epochs):
        decoder.train()
        if pretrain_in_vars != in_vars: channel_adapter.train()
            
        train_loss = []
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)
            
            opt.zero_grad()
            if revin is not None: batch_x = revin(batch_x, mode='norm')
            
            # Channel Adaptation
            if pretrain_in_vars != in_vars:
                batch_x_adapted = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)
            else:
                batch_x_adapted = batch_x
            
            # Encoder forward
            with torch.no_grad():
                if batch_x_adapted.shape[-1] > 512:
                    x_in = batch_x_adapted[:, :, -512:]
                else:
                    x_in = torch.nn.functional.pad(batch_x_adapted, (512 - batch_x_adapted.shape[-1], 0))
                
                # TimeVLM: _process가 [B, C, T]를 기대 / 나머지: [B, 1, C, T]
                if arch_key == "timevlm":
                    emb = encoder._process(x_in, target_res=224, training=True, is_downstream=True)
                elif arch_key in ["patchtst", "utica"]:
                    actual_enc = encoder.encoder if arch_key == "utica" else encoder
                    emb = actual_enc._process(x_in.unsqueeze(1), max_len=512, offsets=None)
                else:
                    emb = encoder._process(x_in.unsqueeze(1), target_res=224, training=True, is_downstream=True)
                
                # TiViT Independent는 [B, C, Embed]를 반환하므로 채널 방향 평균 취함
                if arch_key == "tivit_indep":
                    emb = emb.mean(1)
                
            preds = decoder(emb)
            if revin is not None: preds = revin(preds, mode='denorm')
            
            loss = criterion(preds, batch_y)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            
        avg_train_loss = np.mean(train_loss)
        
        # Validation
        decoder.eval(); 
        if pretrain_in_vars != in_vars: channel_adapter.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)
                if revin is not None: batch_x = revin(batch_x, mode='norm')
                if pretrain_in_vars != in_vars:
                    batch_x_adapted = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)
                else:
                    batch_x_adapted = batch_x
                
                if batch_x_adapted.shape[-1] > 512:
                    x_in = batch_x_adapted[:, :, -512:]
                else:
                    x_in = torch.nn.functional.pad(batch_x_adapted, (512 - batch_x_adapted.shape[-1], 0))
                
                if arch_key == "timevlm":
                    emb = encoder._process(x_in, target_res=224, training=False, is_downstream=True)
                elif arch_key in ["patchtst", "utica"]:
                    actual_enc = encoder.encoder if arch_key == "utica" else encoder
                    emb = actual_enc._process(x_in.unsqueeze(1), max_len=512, offsets=None)
                else:
                    emb = encoder._process(x_in.unsqueeze(1), target_res=224, training=False, is_downstream=True)
                
                if arch_key == "tivit_indep":
                    emb = emb.mean(1)
                preds = decoder(emb)
                if revin is not None: preds = revin(preds, mode='denorm')
                val_loss.append(criterion(preds, batch_y).item())
                
        avg_val_loss = np.mean(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'decoder': decoder.state_dict(), 'adapter': channel_adapter.state_dict() if pretrain_in_vars != in_vars else None}, 
                       os.path.join(args.log_dir, f"best_model_{args.pred_len}.pt"))
            
    # 7. Test Evaluation
    best_ckpt = torch.load(os.path.join(args.log_dir, f"best_model_{args.pred_len}.pt"))
    decoder.load_state_dict(best_ckpt['decoder'])
    if pretrain_in_vars != in_vars and best_ckpt['adapter'] is not None:
        channel_adapter.load_state_dict(best_ckpt['adapter'])
    decoder.eval(); 
    if pretrain_in_vars != in_vars: channel_adapter.eval()
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device, dtype=torch.float32)
            if revin is not None: batch_x = revin(batch_x, mode='norm')
            if pretrain_in_vars != in_vars:
                batch_x_adapted = channel_adapter(batch_x.transpose(1, 2)).transpose(1, 2)
            else:
                batch_x_adapted = batch_x
                
            if batch_x_adapted.shape[-1] > 512:
                x_in = batch_x_adapted[:, :, -512:]
            else:
                x_in = torch.nn.functional.pad(batch_x_adapted, (512 - batch_x_adapted.shape[-1], 0))
            
            if arch_key == "timevlm":
                emb = encoder._process(x_in, target_res=224, training=False, is_downstream=True)
            elif arch_key in ["patchtst", "utica"]:
                actual_enc = encoder.encoder if arch_key == "utica" else encoder
                emb = actual_enc._process(x_in.unsqueeze(1), max_len=512, offsets=None)
            else:
                emb = encoder._process(x_in.unsqueeze(1), target_res=224, training=False, is_downstream=True)
            
            if arch_key == "tivit_indep":
                emb = emb.mean(1)
            preds = decoder(emb)
            if revin is not None: preds = revin(preds, mode='denorm')
            
            preds_np, trues_np = preds.cpu().numpy(), batch_y.cpu().numpy()
            all_preds.append(preds_np)   # [B, C, Pred_len] 누적
            all_trues.append(trues_np)   # [B, C, Pred_len] 누적
    
    # 전체 테스트셋으로 concat 후 최종 MSE/MAE 계산
    all_preds = np.concatenate(all_preds, axis=0)  # [N, C, Pred_len]
    all_trues = np.concatenate(all_trues, axis=0)  # [N, C, Pred_len]
    final_mse = mse(all_trues, all_preds)
    final_mae = mae(all_trues, all_preds)
    
    print("="*50)
    print(f"Zero-Shot Forecasting Test Result ({args.dataset_type}, Pred={args.pred_len})")
    print(f"MSE: {final_mse:.6f} | MAE: {final_mae:.6f}")
    print("="*50)
    
    # 텍스트 파일 저장 (summary)
    summary_path = os.path.join(args.log_dir, "results_summary.txt")
    with open(summary_path, "a") as f:
        f.write(f"Target: {args.target_dataset} | Pretrain: {args.pretrain_dataset} | Pred_len: {args.pred_len:4d} | MSE: {final_mse:.6f} | MAE: {final_mae:.6f}\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="basic")
    p.add_argument("--pretrain_dataset", default="electricity")
    p.add_argument("--target_dataset", default="ETTm1")
    p.add_argument("--dataset_type", default="ETTm1")
    p.add_argument("--data_path", required=True)
    p.add_argument("--pretrain_path", required=True)
    
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--pred_len", type=int, default=96)
    
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4) # Linear Probing Learning rate
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--vit_model", default="vit_small_patch14_dinov2")
    p.add_argument("--use_revin", type=lambda x: x.lower() in ("true", "1", "yes", "none"), default=False)
    
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", default="./outputs/linear_probing")
    
    args = p.parse_args()
    
    # dataset_type과 target_dataset을 일치시킴
    args.dataset_type = args.target_dataset
    
    train(args)
