import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import resample_abs_pos_embed
from torchvision.ops import MLP

class LearnableTimeSeriesToImage(nn.Module):
    """Learnable module to convert time series data into image tensors (TimeVLM Official Code Adapted)"""
    def __init__(self, input_dim, hidden_dim, output_channels, periodicity, image_size=512):
        super(LearnableTimeSeriesToImage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.periodicity = periodicity
        self.image_size = image_size

        # 1D convolutional layer
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D convolution layers
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)

    def forward(self, x_enc):
        """Convert input time series to image tensor [B, output_channels, image_size, image_size]"""
        B, L, D = x_enc.shape

        # Generate periodicity encoding (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)  # [B, L, D, 2]

        # FFT frequency encoding (magnitude)
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # Combine all features: raw + FFT + periodic
        x_enc_expanded = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc_combined = torch.cat([x_enc_expanded, x_fft_mag, periodicity_encoding], dim=-1)  # [B, L, D, 4]

        # Reshape for 1D convolution
        x_enc_combined = x_enc_combined.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc_combined = x_enc_combined.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_conv1d = self.conv1d(x_enc_combined)  # [B*D, hidden_dim, L]
        x_conv1d = x_conv1d.reshape(B, D, self.hidden_dim, L)  # [B, D, hidden_dim, L]

        # 2D Convolution processing
        x_conv2d = x_conv1d.permute(0, 2, 1, 3)  # [B, hidden_dim, D, L]
        x_conv2d = torch.tanh(self.conv2d_1(x_conv2d))
        x_conv2d = torch.tanh(self.conv2d_2(x_conv2d))

        # Resize to fixed base resolution (e.g. 512x512)
        x_out = F.interpolate(x_conv2d, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return x_out  # [B, output_channels, 512, 512]


class TimeVLMEncoder(nn.Module):
    def __init__(self, in_vars: int, model_name: str = "vit_small_patch14_dinov2", proj_dim: int = 128, embed_dim: int = 384):
        """
        Args:
            in_vars: Number of input variables/channels.
            model_name: timm ViT model name.
            proj_dim: Output projection head dimension.
            embed_dim: ViT embedding dimension.
        """
        import random
        self.random = random
        super().__init__()
        
        # 1D to 2D image transformation using TimeVLM's Learnable module
        self.to_img = LearnableTimeSeriesToImage(input_dim=in_vars, hidden_dim=64, output_channels=3, periodicity=24, image_size=512)
        
        # Initialize ViT with 3 input channels instead of 1
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=None, in_chans=3)
        self.patch_size = self.backbone.patch_embed.patch_size[0]
        if hasattr(self.backbone.patch_embed, 'strict_img_size'): 
            self.backbone.patch_embed.strict_img_size = False
            
        self.proj_head = MLP(embed_dim, [embed_dim * 4, embed_dim * 4, proj_dim], norm_layer=nn.BatchNorm1d)

    def _2d_crop(self, x: torch.Tensor, target_res: int, training: bool) -> torch.Tensor:
        """
        2D spatial crop from the 512x512 visual feature manifold.
        """
        _, _, H, W = x.shape
        if training:
            h_start = self.random.randint(0, H - target_res)
            w_start = self.random.randint(0, W - target_res)
        else:
            h_start = (H - target_res) // 2
            w_start = (W - target_res) // 2
        return x[:, :, h_start:h_start + target_res, w_start:w_start + target_res]

    def _process(self, x: torch.Tensor, target_res: int, training: bool = True, is_downstream: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B_v, C, T]
            target_res: Custom target image resolution (224 for global, 98 for local)
        """
        # Convert to [B_v, T, C] specifically for LearnableTimeSeriesToImage [B, L, D]
        x_transposed = x.transpose(1, 2)
        
        # Mapping 1D to 3-Channel 512x512 Visual Representation [B_v, 3, 512, 512]
        img_512 = self.to_img(x_transposed)
        
        # Perform 2D Random / Center Crop or Resize to get [B_v, 3, target_res, target_res]
        if is_downstream:
            img_plane = F.interpolate(img_512, size=(target_res, target_res), mode='bilinear', align_corners=False)
        else:
            img_plane = self._2d_crop(img_512, target_res, training)
        
        # ViT Patch Embedding + Position Encoding Resampling
        orig_pos = self.backbone.pos_embed
        grid_size = (target_res // self.patch_size, target_res // self.patch_size)
        
        x_vit = self.backbone.patch_embed(img_plane)
        if self.backbone.cls_token is not None:
            x_vit = torch.cat((self.backbone.cls_token.expand(x_vit.shape[0], -1, -1), x_vit), dim=1)
            
        pos_embed = resample_abs_pos_embed(
            orig_pos, new_size=grid_size,
            num_prefix_tokens=1 if self.backbone.cls_token is not None else 0
        )
        
        x_vit = self.backbone.norm(self.backbone.blocks(self.backbone.pos_drop(x_vit + pos_embed)))
        
        # Pooling: CLS token or average pooling depending on backbone configuration
        return x_vit[:, 1:].mean(1) if self.backbone.global_pool == 'avg' else x_vit[:, 0]

    def forward(self, views: dict) -> tuple:
        """
        Args:
            views: Dict with 'global' [B, 2, C, 512] and 'local' [B, 6, C, 512].
        Returns:
            Tuple of (all_emb [B, 8, embed_dim], proj [8, B, proj_dim]).
        """
        B = views['global'].shape[0]
        training = self.training
        
        # Global Views: [B, 2, C, 512] -> flatten -> [B*2, C, 512]
        g_emb = self._process(views['global'].flatten(0, 1), 224, training)
        
        # Local Views: [B, 6, C, 512] -> flatten -> [B*6, C, 512]
        l_emb = self._process(views['local'].flatten(0, 1), 98, training)
        
        # Re-group embeddings
        all_emb = torch.cat([g_emb.reshape(B, 2, -1), l_emb.reshape(B, 6, -1)], dim=1)  # [B, 8, embed_dim]
        proj = self.proj_head(all_emb.flatten(0, 1)).reshape(B, 8, -1).transpose(0, 1)   # [8, B, proj_dim]
        
        return all_emb, proj
