import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import timm
from timm.layers import resample_abs_pos_embed

class MultiResViTConvEncoder(nn.Module):
    def __init__(self, in_vars: int, model_name: str = "vit_small_patch14_dinov2", proj_dim: int = 128, embed_dim: int = 384, target_h: int = 512):
        """
        LeJEPA Channel-Independent Conv1D Variant (arch='conv'):
        각 채널을 독립적인 1D 시계열로 간주하고, Conv1D 레이어를 활용하여 512 차원으로 부풀려
        비전 모델을 위한 2D 캔버스로 변환합니다.
        
        Args:
            in_vars: 입력 다변량 채널 갯수.
            model_name: timm ViT 모델 이름.
            proj_dim: 최종 프로젝션 차원.
            embed_dim: ViT 임베딩 차원.
            target_h: Conv1D를 통해 부풀릴 Height (목표) 차원 크기.
        """
        super().__init__()
        self.in_vars = in_vars
        self.target_h = target_h
        
        # [B*C, 1, 512] -> [B*C, 512, 512]로 확장 (1D Conv)
        self.conv_expand = nn.Conv1d(
            in_channels=1, 
            out_channels=target_h,
            kernel_size=3,
            padding=1  # 512 Time Width 보존
        )
        self.norm = nn.BatchNorm1d(target_h)
        self.activation = nn.GELU()
        
        # ViT 입력 채널은 1채널 (Channel Independent / Height Expansion)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=None, in_chans=1)
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        if hasattr(self.backbone.patch_embed, 'strict_img_size'): 
            self.backbone.patch_embed.strict_img_size = False
            
        self.proj_head = MLP(embed_dim, [embed_dim * 4, embed_dim * 4, proj_dim], norm_layer=nn.BatchNorm1d)

    def _2d_crop(self, x: torch.Tensor, target_res: int, training: bool, B_v: int) -> torch.Tensor:
        """2D spatial crop from the 1-Channel 512x512 feature-temporal manifold."""
        # x is [B_v*C, 1, H, W]
        _, _, H, W = x.shape
        C_ch = x.shape[0] // B_v
        
        x_view = x.view(B_v, C_ch, H, W)
        out = torch.empty((B_v, C_ch, target_res, target_res), device=x.device, dtype=x.dtype)
        
        if training:
            for i in range(B_v):
                h_start = random.randint(0, H - target_res)
                w_start = random.randint(0, W - target_res)
                out[i] = x_view[i, :, h_start:h_start + target_res, w_start:w_start + target_res]
        else:
            h_start = (H - target_res) // 2
            w_start = (W - target_res) // 2
            out = x_view[:, :, h_start:h_start + target_res, w_start:w_start + target_res]
            
        return out.view(B_v * C_ch, 1, target_res, target_res)

    def _process(self, x: torch.Tensor, target_res: int, training: bool = True, is_downstream: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B_v, 1, C, T=512] 입력
        """
        # 0. Flatten Channels into Batch Dimension: [B_v, 1, C, T] -> [B_v * C, 1, T]
        # C > 1 조건 제거: C=1 포함 모든 4D 입력을 3D로 Fold해야 Conv1d가 에러 없이 받을 수 있음
        if x.dim() == 4:
            B_v, _, C, T = x.shape
            x = x.transpose(1, 2).reshape(B_v * C, 1, T)
        
        # 1. Conv1D Feature Expansion: [B_v*C, 1, T] -> [B_v*C, 512, T]
        out = self.conv_expand(x)
        out = self.norm(out)
        out = self.activation(out)
        
        # 2. Reshape into 2D canvas shape for ViT: [B_v*C, 512, 512] -> [B_v*C, 1, 512, 512]
        x_manifold = out.unsqueeze(1)
        
        # 3. 2D Spatial Crop or Resize
        if is_downstream:
            x_cropped = F.interpolate(x_manifold, size=(target_res, target_res), mode='bilinear', align_corners=False)
        else:
            x_cropped = self._2d_crop(x_manifold, target_res, training, B_v=B_v)

        # 4. ViT Forward (1-Channel Input, 독립적인 이미지로서 처리)
        orig_pos = self.backbone.pos_embed
        grid_size = (target_res // self.patch_size, target_res // self.patch_size)
        
        x_patches = self.backbone.patch_embed(x_cropped)
        
        if self.backbone.cls_token is not None:
            x_patches = torch.cat((self.backbone.cls_token.expand(x_patches.shape[0], -1, -1), x_patches), dim=1)
            
        pos_embed = resample_abs_pos_embed(
            orig_pos, new_size=grid_size,
            num_prefix_tokens=1 if self.backbone.cls_token is not None else 0
        )
        
        feat = self.backbone.norm(self.backbone.blocks(self.backbone.pos_drop(x_patches + pos_embed)))
        emb = feat[:, 1:].mean(1) if self.backbone.global_pool == 'avg' else feat[:, 0] # [B_v*C, 384]
        
        return emb

    def forward(self, views: dict) -> tuple:
        B = views['global'].shape[0]
        C = views['global'].shape[2]
        training = self.training

        g_in = views['global'].flatten(0, 1).unsqueeze(1) # [B*2, 1, C, 512]
        l_in = views['local'].flatten(0, 1).unsqueeze(1)  # [B*6, 1, C, 512]

        g_emb = self._process(g_in, 224, training) # [B*2*C, 384]
        l_emb = self._process(l_in, 98, training)  # [B*6*C, 384]

        # SIGReg 연산을 위해 다변량 채널축(C)을 샘플 배치축(Batch)으로 편입시킴
        g_emb = g_emb.reshape(B * 2, C, -1).transpose(0, 1).reshape(B * C, 2, -1)
        l_emb = l_emb.reshape(B * 6, C, -1).transpose(0, 1).reshape(B * C, 6, -1)

        all_emb = torch.cat([g_emb, l_emb], dim=1) # [B*C, 8, 384]
        proj = self.proj_head(all_emb.flatten(0, 1)).reshape(B * C, 8, -1).transpose(0, 1) # [8, B*C, proj_dim]
        
        return all_emb, proj
