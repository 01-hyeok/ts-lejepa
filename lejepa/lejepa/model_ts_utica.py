import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

# 기존 PatchTS1DEncoder 재사용
from lejepa.model_ts_lejepa_1d import PatchTS1DEncoder

class UTICAMultiCropGenerator(nn.Module):
    """
    UTICA-style Time-Series Multi-Crop View Generator
    - Batch 단위의 연산을 유지하며 내부적으로 각 샘플별 다른 영역을 Random Crop & Resize
    """
    def __init__(
        self,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_scale: Tuple[float, float] = (0.1, 0.4),
        global_crop_size: int = 512,
        local_crop_size: int = 256,
        num_global_crops: int = 2,
        num_local_crops: int = 6,
        jitter_std_ratio: float = 0.05,
        jitter_prob: float = 0.5,
        scaling_range: Tuple[float, float] = (0.95, 1.05),
        scaling_prob: float = 0.5,
    ):
        super().__init__()
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        
        self.jitter_std_ratio = jitter_std_ratio
        self.jitter_prob = jitter_prob
        self.scaling_range = scaling_range
        self.scaling_prob = scaling_prob

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """독립적인 데이터 증강 적용 (Gaussian Jitter & Amplitude Scaling)"""
        B, C, T = x.shape
        device = x.device
        
        # 1. Gaussian Jitter
        jitter_mask = (torch.rand(B, 1, 1, device=device) < self.jitter_prob).float()
        std = x.std(dim=-1, keepdim=True)
        noise = torch.randn_like(x) * (std * self.jitter_std_ratio)
        x = x + noise * jitter_mask
        
        # 2. Amplitude Scaling
        scaling_mask = (torch.rand(B, 1, 1, device=device) < self.scaling_prob).float()
        scales = torch.empty(B, C, 1, device=device).uniform_(self.scaling_range[0], self.scaling_range[1])
        scales = torch.where(scaling_mask == 1.0, scales, torch.ones_like(scales))
        x = x * scales
        
        return x

    def _generate_crop(
        self, 
        x: torch.Tensor, 
        scale_range: Tuple[float, float], 
        target_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """F.grid_sample을 활용한 Vectorized Random Crop 및 Resize"""
        B, C, T = x.shape
        device = x.device
        
        ratios = torch.empty(B, device=device).uniform_(scale_range[0], scale_range[1])
        crop_lens = (T * ratios).long().clamp(min=2, max=T)
        
        max_starts = T - crop_lens
        raw_starts = (torch.rand(B, device=device) * (max_starts + 1).float()).long()
        starts = torch.min(raw_starts, max_starts).clamp(min=0)
        
        base_grid = torch.linspace(0, 1, steps=target_size, device=device).unsqueeze(0)
        real_indices = starts.unsqueeze(1) + base_grid * (crop_lens.unsqueeze(1) - 1)
        grid_x = (real_indices / (T - 1)) * 2 - 1
        grid_y = torch.zeros_like(grid_x)
        
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)
        x_4d = x.unsqueeze(2)
        cropped_resized = F.grid_sample(x_4d, grid, mode='bilinear', align_corners=True)
        cropped_resized = cropped_resized.squeeze(2) 
        
        return cropped_resized, starts, crop_lens

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        global_views, global_starts, global_lens = [], [], []
        local_views, local_starts, local_lens = [], [], []
        
        for _ in range(self.num_global_crops):
            view, starts, crop_lens = self._generate_crop(x, self.global_crop_scale, self.global_crop_size)
            view = self._apply_augmentation(view)  
            global_views.append(view)           
            global_starts.append(starts)        
            global_lens.append(crop_lens)       
            
        for _ in range(self.num_local_crops):
            view, starts, crop_lens = self._generate_crop(x, self.local_crop_scale, self.local_crop_size)
            view = self._apply_augmentation(view)  
            local_views.append(view)            
            local_starts.append(starts)         
            local_lens.append(crop_lens)        
            
        return {
            "global_views": global_views,
            "local_views": local_views,
            "global_starts": global_starts,
            "global_lens": global_lens,
            "local_starts": local_starts,
            "local_lens": local_lens
        }

class UTICAEncoder(nn.Module):
    """
    UTICA-style Encoder wrapper.
    GPU 내에서 UTICAMultiCropGenerator를 통해 View를 쪼갠 후, PatchTS1DEncoder로 포워딩합니다.
    """
    def __init__(self, in_vars: int, d_model: int = 384, patch_size: int = 16, proj_dim: int = 128, use_revin: bool = True):
        super().__init__()
        # GPU 기반 Multi-Crop 제너레이터 인스턴스화
        self.generator = UTICAMultiCropGenerator()
        
        # Backbone으로 쓰일 1D Patch Encoder
        self.encoder = PatchTS1DEncoder(
            in_vars=in_vars,
            d_model=d_model,
            patch_size=patch_size,
            n_heads=6,
            n_layers=8,
            proj_dim=proj_dim,
            dropout=0.1,
            use_revin=use_revin
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [Batch, Channels, Time] 형태의 원본(가장 긴) 텐서 입력
        Returns:
            all_emb, proj (PatchTS1DEncoder 의 반환 규격과 동일)
        """
        # 1. 뷰 생성 트리거
        views_dict = self.generator(x)
        
        # 2. Backbone 네트워크가 원하는 규격의 Dictionary 로 묶기
        # generator 반환은 List[Tensor] 이므로 이를 stack 하여 [B, n_views, C, T] 형태의 Tensor 로 변환해줍니다.
        global_stacked = torch.stack(views_dict["global_views"], dim=1)  # [B, 2, C, 512]
        local_stacked = torch.stack(views_dict["local_views"], dim=1)    # [B, 6, C, 256]
        
        global_offsets = torch.stack(views_dict["global_starts"], dim=1) # [B, 2]
        global_lengths = torch.stack(views_dict["global_lens"], dim=1)   # [B, 2]
        local_offsets = torch.stack(views_dict["local_starts"], dim=1)   # [B, 6]
        local_lengths = torch.stack(views_dict["local_lens"], dim=1)     # [B, 6]
        
        encoder_input = {
            "global": global_stacked,
            "local": local_stacked,
            "global_offsets": global_offsets,
            "global_lengths": global_lengths,
            "local_offsets": local_offsets,
            "local_lengths": local_lengths
        }

        
        # 3. Backbone 인코더 수행
        return self.encoder(encoder_input)
