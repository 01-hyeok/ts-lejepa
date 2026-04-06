import os
import random
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler
    from torch import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

# PyTorch 스케줄러 관련 불필요한 경고 억제
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

from lejepa.data_ts_lejepa_basic import get_1d_multires_loaders
from lejepa.model_ts_lejepa_basic import MultiResViTEncoder
from lejepa.model_ts_lejepa_tiling import MultiResViTTilingEncoder
from lejepa.model_ts_tivit import TiViTIndependentEncoder, TiViTDependentEncoder
from lejepa.model_ts_timevlm import TimeVLMEncoder
from lejepa.model_ts_conv2d import Conv2DLearnableEncoder
from lejepa.model_ts_lejepa_basic import SIGReg
from lejepa.model_ts_lejepa_ci import MultiResViTCIEncoder
from lejepa.model_ts_conv import MultiResViTConvEncoder
from lejepa.model_ts_lejepa_1d import PatchTS1DEncoder
from lejepa.model_ts_utica import UTICAEncoder
from lejepa.model_ts_timesblock import LeJEPATimesModel

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def compute_collapse_metrics(z: torch.Tensor, prefix: str = "encoder") -> dict:
    """Compute representation collapse diagnostics for TensorBoard logging.
    Args:
        z: Batch of representations from the encoder, shape [Batch, Dim]
        prefix: Prefix string for TensorBoard key naming.
    Returns:
        Dictionary of scalar metrics.
    """
    metrics = {}
    z = z.detach().float()  # Gradient tracking not needed
    
    # 3D 입력 [Batch, Patches/Views, Dim] → [Batch*Patches, Dim] 로 flatten
    if z.dim() == 3:
        z = z.reshape(-1, z.size(-1))  # [Batch*Patches, Dim]
    
    # 1. Feature-wise standard deviation (VICReg style)
    std_per_dim = z.std(dim=0)                                        # [Dim]
    metrics[f"{prefix}/feature_std_mean"] = std_per_dim.mean().item()
    metrics[f"{prefix}/feature_std_min"] = std_per_dim.min().item()
    
    # 2. Dead Dimensions ratio (variance near zero)
    dead_dim_ratio = (std_per_dim < 0.01).float().mean().item()
    metrics[f"{prefix}/dead_dim_ratio"] = dead_dim_ratio
    
    # 3. Effective Rank (Shannon entropy of singular values)
    _, s, _ = torch.linalg.svd(z - z.mean(dim=0, keepdim=True), full_matrices=False)
    s = s.clamp(min=1e-8)
    s_prob = s / s.sum()
    entropy = -(s_prob * torch.log(s_prob)).sum()
    metrics[f"{prefix}/effective_rank"] = torch.exp(entropy).item()
    
    # 4. Average Cosine Similarity within the batch
    if z.size(0) > 1:
        z_norm = F.normalize(z, dim=-1)                               # [N, Dim]
        # .T 대신 .mT 또는 명시적 transpose 사용 (3D deprecation 방지)
        cos_sim_matrix = z_norm @ z_norm.transpose(0, 1)             # [N, N]
        mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        metrics[f"{prefix}/avg_cosine_similarity"] = cos_sim_matrix[mask].mean().item()
    else:
        metrics[f"{prefix}/avg_cosine_similarity"] = 0.0
        
    return metrics

def compute_loss(inv: torch.Tensor, sig: torch.Tensor, args) -> torch.Tensor:
    """Compute total loss based on provided hyperparameters.

    Supports two modes:
    - Lambda mode (legacy): ``loss = (1 - λ) * L_pred + λ * L_sigreg``
    - Alpha/Beta mode:      ``loss = α * L_pred + β * L_sigreg``

    Args:
        inv: Prediction (invariance) loss scalar.
        sig: SIGReg loss scalar.
        args: Parsed argument namespace containing lamb / alpha / beta.
    Returns:
        Total loss scalar.
    """
    # alpha와 beta가 둘 다 명시적으로 전달된 경우 → 독립 가중치 모드
    if args.alpha is not None and args.beta is not None:
        return args.alpha * inv + args.beta * sig
    # lamb가 명시적으로 전달된 경우 → 기존 (1-λ)P + λS 모드
    elif args.lamb is not None:
        return (1 - args.lamb) * inv + args.lamb * sig
    else:
        raise ValueError(
            "loss 가중치를 지정해주세요: --lamb 또는 --alpha + --beta"
        )


def validate(net, sigreg, loader, device, args):
    net.eval(); v_inv, v_sig, v_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for views in loader:
            if isinstance(views, dict):
                for k in views: views[k] = views[k].to(device, non_blocking=True)
            else:
                views = views.to(device, non_blocking=True)
                
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                _, proj = net(views)
                inv = (proj.mean(0) - proj).square().mean()
                sig = sigreg(proj)
                loss = compute_loss(inv, sig, args)
            v_inv += inv.item(); v_sig += sig.item(); v_loss += loss.item()
    return v_inv/len(loader), v_sig/len(loader), v_loss/len(loader)

def train(args):
    set_seed(args.seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # arch를 추가로 넘겨서 UTICA일 경우 transform=None 으로 처리되게 함
    print(args.data_path)
    train_loader, val_loader = get_1d_multires_loaders(args.dataset_type, args.data_path, args.batch_size, args.seq_len, args.stride, args.num_workers, args.max_files, local_len=args.local_len, arch=args.arch)
    
    # 아키텍처별 로그/저장 경로 자동 분리
    args.save_dir = os.path.join(args.save_dir, args.arch)
    args.log_dir = os.path.join(args.log_dir, args.arch)
    
    t_writer = SummaryWriter(os.path.join(args.log_dir, 'train')); v_writer = SummaryWriter(os.path.join(args.log_dir, 'val'))
    
    # 데이터셋 정보 상세 출력
    print(f"\n" + "="*50)
    print(f"🚀 Architecture: {args.arch.upper()}")
    print(f"📊 Dataset Statistics:")
    print(f"   - Sequence Length (T): {args.seq_len}")
    print(f"   - Train Samples: {len(train_loader.dataset)}, Steps: {len(train_loader)}")
    print(f"   - Val Samples: {len(val_loader.dataset)}, Steps: {len(val_loader)}")
    print("="*50 + "\n")
    
    # in_vars 안전하게 추출
    try:
        sample = next(iter(train_loader))
        if isinstance(sample, dict):
            in_vars = sample['global'].shape[2]
        else:
            in_vars = sample.shape[1]  # [B, C, T]
    except (StopIteration, IndexError):
        # 만약 train_loader가 비어있다면 val_loader에서 시도
        sample = next(iter(val_loader))
        if isinstance(sample, dict):
            in_vars = sample['global'].shape[2]
        else:
            in_vars = sample.shape[1]
    
    # 아키텍처 이름 유연화: 하이픈(-)과 언더바(_) 모두 허용
    arch_key = args.arch.lower().replace('-', '_')
    
    if arch_key == "basic":
        net = MultiResViTEncoder(in_vars, args.vit_model, args.proj_dim).to(device)
    elif arch_key in ["tiling", "tiling_repeat"]:
        net = MultiResViTTilingEncoder(in_vars, args.vit_model, args.proj_dim).to(device)
    elif arch_key == "tivit_indep":
        net = TiViTIndependentEncoder(in_vars, args.vit_model, args.proj_dim).to(device)
    elif arch_key == "tivit_dep":
        net = TiViTDependentEncoder(in_vars, args.vit_model, args.proj_dim).to(device)
    elif arch_key == "timevlm":
        net = TimeVLMEncoder(in_vars, args.vit_model, args.proj_dim).to(device)
    elif arch_key == "conv2d":
        net = Conv2DLearnableEncoder(in_vars, args.vit_model, args.proj_dim).to(device)
    elif arch_key == "tiling_ci":
        net = MultiResViTCIEncoder(in_vars, args.vit_model, args.proj_dim).to(device)
    elif arch_key == "conv":
        net = MultiResViTConvEncoder(in_vars, args.vit_model, args.proj_dim).to(device)
    elif arch_key == "patchtst":
        net = PatchTS1DEncoder(in_vars, proj_dim=args.proj_dim, patch_size=args.patch_size, use_revin=args.use_revin).to(device)
    elif arch_key == "utica":
        net = UTICAEncoder(in_vars, proj_dim=args.proj_dim, patch_size=args.patch_size, use_revin=args.use_revin).to(device)
    elif arch_key == "timesnet":
        net = LeJEPATimesModel(in_channels=in_vars, d_model=128, proj_dim=args.proj_dim).to(device)
    elif arch_key == "timesnet_update":
        net = LeJEPATimesModel(in_channels=in_vars, d_model=128, proj_dim=args.proj_dim, norm_type="instance").to(device)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
        
    sigreg = SIGReg().to(device)
    
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-2)
    warmup = len(train_loader)
    scheduler = SequentialLR(opt, [LinearLR(opt, 0.01, total_iters=warmup), CosineAnnealingLR(opt, len(train_loader)*args.epochs - warmup, 1e-6)], [warmup])
    scaler = GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.save_dir, exist_ok=True)
    best_vl = float('inf'); best_vi = float('inf'); best_vsig = float('inf')
    for epoch in range(args.epochs):
        net.train(); t_inv, t_sig, t_loss = 0.0, 0.0, 0.0
        for i, views in enumerate(train_loader):
            if isinstance(views, dict):
                for k in views: views[k] = views[k].to(device, non_blocking=True)
            else:
                views = views.to(device, non_blocking=True)
                
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                repr_out, proj = net(views)
                inv = (proj.mean(0) - proj).square().mean(); sig = sigreg(proj)
                loss = compute_loss(inv, sig, args)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); scheduler.step()
            t_inv += inv.item(); t_sig += sig.item(); t_loss += loss.item()
            if (i+1) % args.log_int == 0:
                step = epoch * len(train_loader) + i
                t_writer.add_scalar("Step/Total", loss.item(), step); t_writer.add_scalar("Step/Pred", inv.item(), step); t_writer.add_scalar("Step/SIG", sig.item(), step)
                
                # collapse metrics
                # repr_out: [Batch, Patches, Dim] → 패치 평균으로 [Batch, Dim] 변환
                repr_pooled = repr_out.mean(dim=1) if repr_out.dim() == 3 else repr_out
                repr_metrics = compute_collapse_metrics(repr_pooled, prefix="Train/Collapse_Repr")
                proj_metrics = compute_collapse_metrics(proj, prefix="Train/Collapse_Proj")
                for k, v in {**repr_metrics, **proj_metrics}.items():
                    t_writer.add_scalar(k, v, step)
        
        ai, asig, al = t_inv/len(train_loader), t_sig/len(train_loader), t_loss/len(train_loader)
        vi, vsig, vl = validate(net, sigreg, val_loader, device, args)
        
        # 상세 에폭 로깅 (사용자 요청 지표 전체 반영)
        cur_lr = opt.param_groups[0]['lr']
        log_str = (f"Epoch {epoch+1}/{args.epochs} | "
                   f"Steps: {len(train_loader)} | "
                   f"LR: {cur_lr:.6f} | "
                   f"Train Loss: {al:.6f} (P:{ai:.4f}, S:{asig:.4f}) | "
                   f"Val Loss: {vl:.6f} (P:{vi:.4f}, S:{vsig:.4f})")
        print(log_str)
        t_writer.add_scalar("Loss/Total", al, epoch); t_writer.add_scalar("Loss/Pred", ai, epoch); t_writer.add_scalar("Loss/SIG", asig, epoch)
        v_writer.add_scalar("Loss/Total", vl, epoch); v_writer.add_scalar("Loss/Pred", vi, epoch); v_writer.add_scalar("Loss/SIG", vsig, epoch)
        
        state = {'state_dict': net.state_dict(), 'epoch': epoch, 'val_loss': vl, 'val_pred': vi, 'val_sig': vsig}
        if vl < best_vl:
            best_vl = vl; torch.save(state, os.path.join(args.save_dir, f"lejepa_best_total_{args.dataset_type}.pt"))
        if vi < best_vi:
            best_vi = vi; torch.save(state, os.path.join(args.save_dir, f"lejepa_best_pred_{args.dataset_type}.pt"))
        if vsig < best_vsig:
            best_vsig = vsig; torch.save(state, os.path.join(args.save_dir, f"lejepa_best_sig_{args.dataset_type}.pt"))
    t_writer.close(); v_writer.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_type", default="electricity"); p.add_argument("--data_path", required=True); p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32); p.add_argument("--seq_len", type=int, default=512); p.add_argument("--stride", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--lr", type=float, default=2e-4)
    # loss 가중치: --lamb 단독 OR --alpha + --beta 쌍으로 전달
    # lamb 모드  → (1-λ)*L_pred + λ*L_sigreg  (기존 방식)
    # alpha/beta → α*L_pred + β*L_sigreg       (독립 가중치, 신규)
    p.add_argument("--lamb",  type=float, default=None, help="SIGReg 가중치 λ: (1-λ)P + λS 방식 (구버전 호환)")
    p.add_argument("--alpha", type=float, default=None, help="prediction loss 가중치 α (--beta와 함께 사용)")
    p.add_argument("--beta",  type=float, default=None, help="sigreg loss 가중치 β (--alpha와 함께 사용)")
    p.add_argument("--proj_dim", type=int, default=128); p.add_argument("--vit_model", default="vit_small_patch14_dinov2")
    p.add_argument("--patch_size", type=int, default=16, help="patchtst arch 전용: 1D 패치 크기 (시간 프레임)")
    p.add_argument("--local_len", type=int, default=256, help="Local View 크롭 길이 (타임스텝 단위): 256 = patch_size * 16")
    p.add_argument("--num_workers", type=int, default=8); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", default="basic")
    p.add_argument("--use_revin", type=lambda x: x.lower() in ("true", "1", "yes"), default=True, help="RevIN(Instance Norm) 적용 여부 (True/False)")
    p.add_argument("--save_dir", default="./outputs/pretrain"); p.add_argument("--log_dir", default="./logs/pretrain"); p.add_argument("--log_int", type=int, default=20)
    train(p.parse_args())
