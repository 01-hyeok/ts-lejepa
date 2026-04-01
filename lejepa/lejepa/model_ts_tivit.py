import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import timm
from timm.layers import resample_abs_pos_embed
from einops import rearrange

class TiViTIndependentEncoder(nn.Module):
    def __init__(self, in_vars: int, model_name: str = "vit_small_patch14_dinov2", proj_dim: int = 128, embed_dim: int = 384):
        """
        TiViT Channel Independent:
        각 채널을 개별적인 1채널 이미지로 변환하여 처리합니다.
        """
        super().__init__()
        self.in_vars = in_vars
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=None, in_chans=1)
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        if hasattr(self.backbone.patch_embed, 'strict_img_size'): self.backbone.patch_embed.strict_img_size = False
        self.proj_head = MLP(embed_dim, [embed_dim * 4, embed_dim * 4, proj_dim], norm_layer=nn.BatchNorm1d)

    def _2d_crop(self, x: torch.Tensor, target_res: int, training: bool, B_v: int) -> torch.Tensor:
        """512x512 매니폴드에서 target_res만큼 크롭합니다."""
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
            x: [B_v, 1, C, T]
        """
        B_v, _, C, T = x.shape
        seg_len = 16 # 512의 약수 (rearrange용)
        patch_size = self.patch_size # 14 (ViT 패치용)
        
        # 1. 512x512 전체 매니폴드(Canvas) 생성
        # Segment + Stacking logic
        # T=512를 패치 크기로 쪼개서 직사각형 준비
        target_len = 512 # 매니폴드 기준 길이를 512로 고정
        if T > target_len:
            x = x[:, :, :, -target_len:]
        elif T < target_len:
            x = F.pad(x, (target_len - T, 0))
            
        # [B_v, 1, C, 512] -> [B_v*C, 1, Np, seg_len]
        x = rearrange(x, 'b 1 c (np p) -> (b c) 1 np p', p=seg_len)
        x = F.interpolate(x, size=(512, 512), mode='nearest') # [B_v*C, 1, 512, 512]

        # 2. 2D Crop (224 or 98) or Resize
        # 이제 target_res는 14의 배수이므로 안전함
        if is_downstream:
            x = F.interpolate(x, size=(target_res, target_res), mode='bilinear', align_corners=False)
        else:
            x = self._2d_crop(x, target_res, training, B_v=B_v)

        # 3. ViT Forward
        orig_pos = self.backbone.pos_embed
        grid_size = (target_res // patch_size, target_res // patch_size)
        x = self.backbone.patch_embed(x)
        if self.backbone.cls_token is not None:
            x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = resample_abs_pos_embed(
            orig_pos, new_size=grid_size,
            num_prefix_tokens=1 if self.backbone.cls_token is not None else 0
        )
        x = self.backbone.norm(self.backbone.blocks(self.backbone.pos_drop(x + pos_embed)))
        emb = x[:, 1:].mean(1) if self.backbone.global_pool == 'avg' else x[:, 0]
        
        return rearrange(emb, '(b c) d -> b c d', b=B_v)

    def forward(self, views: dict) -> tuple:
        B = views['global'].shape[0]
        training = self.training

        # Global: 224, Local: 98 (사전 학습된 가중치와 정합성 유지)
        g_emb = self._process(views['global'].flatten(0, 1).unsqueeze(1), 224, training)
        l_emb = self._process(views['local'].flatten(0, 1).unsqueeze(1), 98, training)

        all_emb = torch.cat([g_emb.reshape(B, 2, self.in_vars, -1), 
                            l_emb.reshape(B, 6, self.in_vars, -1)], dim=1)
        agg_emb = all_emb.mean(2)
        proj = self.proj_head(agg_emb.flatten(0, 1)).reshape(B, 8, -1).transpose(0, 1)
        return agg_emb, proj

class TiViTDependentEncoder(nn.Module):
    def __init__(self, in_vars: int, model_name: str = "vit_small_patch14_dinov2", proj_dim: int = 128, embed_dim: int = 384):
        """
        TiViT Channel Dependent:
        모든 채널을 ViT의 입력 채널(in_chans=C)로 한꺼번에 처리합니다.
        """
        super().__init__()
        self.in_vars = in_vars
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=None, in_chans=in_vars)
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        if hasattr(self.backbone.patch_embed, 'strict_img_size'): self.backbone.patch_embed.strict_img_size = False
        self.proj_head = MLP(embed_dim, [embed_dim * 4, embed_dim * 4, proj_dim], norm_layer=nn.BatchNorm1d)

    def _2d_crop(self, x: torch.Tensor, target_res: int, training: bool) -> torch.Tensor:
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
        B_v, _, C, T = x.shape
        seg_len = 16 # 512의 약수 (rearrange용)
        patch_size = self.patch_size # 14 (ViT 패치용)
        
        # 1. 512x512 매니폴드 생성
        target_len = 512
        if T > target_len:
            x = x[:, :, :, -target_len:]
        elif T < target_len:
            x = F.pad(x, (target_len - T, 0))
            
        x = rearrange(x, 'b 1 c (np p) -> b c np p', p=seg_len)
        x = F.interpolate(x, size=(512, 512), mode='nearest')

        # 2. 2D Crop (224 or 98) or Resize
        if is_downstream:
            x = F.interpolate(x, size=(target_res, target_res), mode='bilinear', align_corners=False)
        else:
            x = self._2d_crop(x, target_res, training)

        # 3. ViT Forward
        orig_pos = self.backbone.pos_embed
        grid_size = (target_res // patch_size, target_res // patch_size)
        x = self.backbone.patch_embed(x)
        if self.backbone.cls_token is not None:
            x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = resample_abs_pos_embed(
            orig_pos, new_size=grid_size,
            num_prefix_tokens=1 if self.backbone.cls_token is not None else 0
        )
        x = self.backbone.norm(self.backbone.blocks(self.backbone.pos_drop(x + pos_embed)))
        emb = x[:, 1:].mean(1) if self.backbone.global_pool == 'avg' else x[:, 0]
        return emb

    def forward(self, views: dict) -> tuple:
        B = views['global'].shape[0]
        training = self.training

        g_emb = self._process(views['global'].flatten(0, 1).unsqueeze(1), 224, training)
        l_emb = self._process(views['local'].flatten(0, 1).unsqueeze(1), 98, training)

        all_emb = torch.cat([g_emb.reshape(B, 2, -1), l_emb.reshape(B, 6, -1)], dim=1)
        proj = self.proj_head(all_emb.flatten(0, 1)).reshape(B, 8, -1).transpose(0, 1)
        return all_emb, proj
