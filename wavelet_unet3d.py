"""
3D UNet with wavelet-based cross-scale attention for abdominal multi-organ CT segmentation.

The network is tailored for 13 abdominal organs (excluding background) and operates on
96x96x96 input patches. The output is raw logits with 13 channels.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from utils import DWT3D


class PatchEmbedding(nn.Module):
    """Embed high-frequency wavelet components into patch tokens."""

    def __init__(self, patch_size: Tuple[int, int, int], img_size: Tuple[int, int, int], in_channels: int, out_channels: int):
        super().__init__()
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.patch_embeddings = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size, stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, out_channels))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class PoolEmbedding(nn.Module):
    """Embed encoder skip features using average pooling tokens."""

    def __init__(self, patch_size: Tuple[int, int, int], img_size: Tuple[int, int, int], in_channels: int):
        super().__init__()
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.pool_embedding = nn.AvgPool3d(kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool_embedding(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class WCSA(nn.Module):
    """Wavelet Cross-Scale Attention block with decoupled channel/spatial attention."""

    def __init__(self, in_channels: int, hidden_size: int, num_heads: int = 4, channel_attn_drop: float = 0.1,
                 spatial_attn_drop: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)

        self.q_c = nn.Linear(in_channels, in_channels, bias=False)
        self.q_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_s = nn.Linear(in_channels // 4, in_channels // 4, bias=False)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward(self, s: torch.Tensor, h: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
        B, N, C = sh.shape
        C_c = s.shape[2]
        C_s = h.shape[2]

        q_c = self.q_c(s).reshape(B, self.num_heads, N, C_c // self.num_heads).transpose(-1, -2)
        q_s = self.q_s(sh).reshape(B, self.num_heads, N, C // self.num_heads)
        k_c = self.k_c(sh).reshape(B, self.num_heads, N, C // self.num_heads)
        v_c = self.v_c(sh).reshape(B, self.num_heads, N, C // self.num_heads)
        k_s = self.k_s(sh).reshape(B, self.num_heads, N, C // self.num_heads)
        v_s = self.v_s(h).reshape(B, self.num_heads, N, C_s // self.num_heads)

        attn_ca = (q_c @ k_c) / math.sqrt(k_c.shape[-1]) * self.temperature
        attn_ca = self.attn_drop(attn_ca.softmax(dim=-1))
        x_ca = (attn_ca @ v_c.transpose(-2, -1)).reshape(B, N, C_c)

        attn_sa = (q_s @ k_s.transpose(-2, -1)) / math.sqrt(k_s.shape[-1]) * self.temperature2
        attn_sa = self.attn_drop_2(attn_sa.softmax(dim=-1))
        x_sa = (attn_sa @ v_s).reshape(B, N, C_s)

        x = torch.cat([x_ca, x_sa], dim=2)
        return x


class TransformerBlock(nn.Module):
    """Transformer block integrating wavelet cross-scale attention."""

    def __init__(
        self,
        skip_channels: int,
        num_heads: int,
        patch_size: Tuple[int, int, int],
        img_size: Tuple[int, int, int],
        hf_channels: int = 7,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if skip_channels % num_heads != 0:
            raise ValueError("skip_channels should be divisible by num_heads.")

        hf_embed_channels = skip_channels // 4
        hidden_size = skip_channels + hf_embed_channels

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size, img_size=img_size, in_channels=hf_channels, out_channels=hf_embed_channels
        )
        self.pool_embedding = PoolEmbedding(patch_size=patch_size, img_size=img_size, in_channels=skip_channels)
        self.norm_s = nn.LayerNorm(skip_channels)
        self.norm_h = nn.LayerNorm(hf_embed_channels)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.wcsa_block = WCSA(
            in_channels=skip_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
        )
        self.conv = nn.Sequential(
            nn.Conv3d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.recon = nn.ConvTranspose3d(hidden_size, hidden_size, kernel_size=patch_size, stride=patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    def forward(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        s = self.pool_embedding(s)
        h = self.patch_embedding(h)

        s = self.norm_s(s)
        h = self.norm_h(h)

        sh = torch.cat([s, h], dim=2)

        B, N, C = sh.shape
        H, W, D = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
            self.img_size[2] // self.patch_size[2],
        )

        attn = sh + self.wcsa_block(s, h, sh) * self.gamma

        attn = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        x = self.conv(attn)
        x = self.recon(x)
        return x


class ConvBlock(nn.Module):
    """Two-layer 3D convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block returning features and pooled output."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution upsampling and skip concatenation."""

    def __init__(self, in_channels: Tuple[int, int, int], out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels[0], in_channels[0], kernel_size=2, stride=2)
        self.conv = ConvBlock(sum(in_channels), out_channels)

    def forward(self, x: torch.Tensor, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, a, s], dim=1)
        x = self.conv(x)
        return x


class WaveletUNet3D(nn.Module):
    """Wavelet-augmented 3D UNet for 13-organ abdominal CT segmentation."""

    def __init__(self):
        super().__init__()

        self.dwt = DWT3D()

        self.e1 = EncoderBlock(1, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)

        self.attention1 = TransformerBlock(
            skip_channels=64,
            num_heads=4,
            patch_size=(8, 8, 8),
            img_size=(48, 48, 48),
            hf_channels=7,
        )
        self.attention2 = TransformerBlock(
            skip_channels=128,
            num_heads=4,
            patch_size=(4, 4, 4),
            img_size=(24, 24, 24),
            hf_channels=7,
        )
        self.attention3 = TransformerBlock(
            skip_channels=256,
            num_heads=4,
            patch_size=(2, 2, 2),
            img_size=(12, 12, 12),
            hf_channels=7,
        )

        self.b1 = ConvBlock(256, 512)

        self.d1 = DecoderBlock((512, 320, 256), 256)
        self.d2 = DecoderBlock((256, 160, 128), 128)
        self.d3 = DecoderBlock((128, 80, 64), 64)
        self.d4 = DecoderBlock((64, 32, 32), 32)

        self.output = nn.Conv3d(32, 13, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1, h1 = self.dwt(x)
        h1 = h1.flatten(1, 2)
        l2, h2 = self.dwt(l1)
        h2 = h2.flatten(1, 2)
        l3, h3 = self.dwt(l2)
        h3 = h3.flatten(1, 2)

        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        a1 = self.attention1(s2, h1)
        a2 = self.attention2(s3, h2)
        a3 = self.attention3(s4, h3)

        b1 = self.b1(p4)

        d1 = self.d1(b1, a3, s4)
        d2 = self.d2(d1, a2, s3)
        d3 = self.d3(d2, a1, s2)
        d4 = self.d4(d3, s1, s1)

        out = self.output(d4)
        return out
