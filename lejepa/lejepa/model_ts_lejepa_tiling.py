import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import timm
from timm.layers import resample_abs_pos_embed

class MultiResViTTilingEncoder(nn.Module):
    def __init__(self, in_vars: int, model_name: str = "vit_small_patch14_dinov2", proj_dim: int = 128, embed_dim: int = 384):
        """
        LeJEPA Tiling Variant (Multi-Channel):
        각 채널을 512x512 매니폴드로 확장한 후, 모든 채널을 ViT의 입력 채널(in_chans=C)로 직접 전달합니다.
        
        Args:
            in_vars: 입력 변수 수 (Channels). ViT의 in_chans로 사용됩니다.
            model_name: timm ViT 모델 이름.
            proj_dim: 최종 프로젝션 차원.
            embed_dim: ViT 임베딩 차원.
        """
        super().__init__()
        self.in_vars = in_vars
        
        # [B, C, 1, 512]의 Height(1)를 512로 선형 투영하여 512x512 매니폴드 생성
        self.height_proj = nn.Linear(1, 512)
        
        # ViT 입력 채널을 데이터셋의 변수 수(in_vars)와 일치시킵니다.
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=None, in_chans=in_vars)
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        if hasattr(self.backbone.patch_embed, 'strict_img_size'): self.backbone.patch_embed.strict_img_size = False
        self.proj_head = MLP(embed_dim, [embed_dim * 4, embed_dim * 4, proj_dim], norm_layer=nn.BatchNorm1d)

    def _2d_crop(self, x: torch.Tensor, target_res: int, training: bool) -> torch.Tensor:
        """
        2D spatial crop from the Multi-Channel 512x512 feature-temporal manifold.
        Args:
            x: [B_v, C, 512, 512]
        """
        B_v, C_ch, H, W = x.shape
        out = torch.empty((B_v, C_ch, target_res, target_res), device=x.device, dtype=x.dtype)
        if training:
            for i in range(B_v):
                h_start = random.randint(0, H - target_res)
                w_start = random.randint(0, W - target_res)
                out[i] = x[i, :, h_start:h_start + target_res, w_start:w_start + target_res]
        else:
            h_start = (H - target_res) // 2
            w_start = (W - target_res) // 2
            out = x[:, :, h_start:h_start + target_res, w_start:w_start + target_res]
        return out

    def _process(self, x: torch.Tensor, target_res: int, training: bool = True, is_downstream: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B_v, 1, C, T=512]
        """
        B_v, _, C, T = x.shape
        
        # 1. 차원 확장 및 투영: [B_v, C, 512] -> [B_v, C, 512, 512]
        # x: [B_v, 1, C, 512] -> squeeze -> [B_v, C, 512] -> unscreen -> [B_v, C, 512, 1]
        x = x.squeeze(1).unsqueeze(-1) # [B_v, C, 512, 1]
        x_manifold = self.height_proj(x) # [B_v, C, 512, 512]
        
        # 2. 2D Spatial Crop or Resize
        # 모든 채널에 대해 동일한 영역을 크롭합니다.
        if is_downstream:
            x_cropped = F.interpolate(x_manifold, size=(target_res, target_res), mode='bilinear', align_corners=False)
        else:
            x_cropped = self._2d_crop(x_manifold, target_res, training) # [B_v, C, target_res, target_res]

        # 3. ViT Forward (Multi-Channel Input)
        orig_pos = self.backbone.pos_embed
        grid_size = (target_res // self.patch_size, target_res // self.patch_size)
        
        # Conv2d patch_embed가 C개 채널을 한꺼번에 처리합니다.
        x_patches = self.backbone.patch_embed(x_cropped)
        
        if self.backbone.cls_token is not None:
            x_patches = torch.cat((self.backbone.cls_token.expand(x_patches.shape[0], -1, -1), x_patches), dim=1)
            
        pos_embed = resample_abs_pos_embed(
            orig_pos, new_size=grid_size,
            num_prefix_tokens=1 if self.backbone.cls_token is not None else 0
        )
        
        feat = self.backbone.norm(self.backbone.blocks(self.backbone.pos_drop(x_patches + pos_embed)))
        emb = feat[:, 1:].mean(1) if self.backbone.global_pool == 'avg' else feat[:, 0] # [B_v, 384]
        
        return emb

    def forward(self, views: dict) -> tuple:
        B = views['global'].shape[0]
        training = self.training

        # Global/Local 뷰를 각각 처리 (Batch 차원으로 뷰 병합)
        g_emb = self._process(views['global'].flatten(0, 1).unsqueeze(1), 224, training)
        l_emb = self._process(views['local'].flatten(0, 1).unsqueeze(1), 98, training)

        all_emb = torch.cat([g_emb.reshape(B, 2, -1), l_emb.reshape(B, 6, -1)], dim=1) # [B, 8, 384]
        proj = self.proj_head(all_emb.flatten(0, 1)).reshape(B, 8, -1).transpose(0, 1)
        return all_emb, proj
