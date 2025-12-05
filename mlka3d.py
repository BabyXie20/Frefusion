"""3D adaptations of LayerNorm and Multi-Scale Large Kernel Attention (MLKA).

These modules mirror a common 2D MLKA design but operate on 5D tensors
``(B, C, D, H, W)`` using 3D convolutions while preserving spatial dimensions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization that supports channels-first 3D tensors.

    Parameters
    ----------
    normalized_shape : int
        Number of feature channels. For channels-last inputs, this is passed to
        :func:`torch.nn.functional.layer_norm`. For channels-first inputs, the
        normalization is computed manually across channels.
    eps : float, optional
        Small constant to avoid division by zero.
    data_format : str, optional
        Either ``"channels_last"`` (``B, D, H, W, C``) or ``"channels_first"``
        (``B, C, D, H, W``). Only these two formats are supported.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class MLKA(nn.Module):
    """Multi-Scale Large Kernel Attention adapted for 3D feature maps.

    The design splits channels into thirds and applies three parallel large
    receptive-field branches using depthwise 3D convolutions. All convolutions
    preserve the input spatial dimensions.
    """

    def __init__(self, n_feats: int):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats must be divisible by 3 for MLKA.")
        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format="channels_first")

        self.proj_first = nn.Sequential(nn.Conv3d(n_feats, i_feats, kernel_size=1, stride=1, padding=0))

        self.LKA3 = nn.Sequential(
            nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=3, stride=1, padding=1, groups=n_feats // 3),
            nn.Conv3d(
                n_feats // 3,
                n_feats // 3,
                kernel_size=5,
                stride=1,
                padding=(5 // 2) * 2,
                dilation=2,
                groups=n_feats // 3,
            ),
            nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=1, stride=1, padding=0),
        )
        self.X3 = nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=3, stride=1, padding=1, groups=n_feats // 3)

        self.LKA5 = nn.Sequential(
            nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=5, stride=1, padding=5 // 2, groups=n_feats // 3),
            nn.Conv3d(
                n_feats // 3,
                n_feats // 3,
                kernel_size=7,
                stride=1,
                padding=(7 // 2) * 3,
                dilation=3,
                groups=n_feats // 3,
            ),
            nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=1, stride=1, padding=0),
        )
        self.X5 = nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=5, stride=1, padding=5 // 2, groups=n_feats // 3)

        self.X7 = nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=7, stride=1, padding=7 // 2, groups=n_feats // 3)
        self.LKA7 = nn.Sequential(
            nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=7, stride=1, padding=7 // 2, groups=n_feats // 3),
            nn.Conv3d(
                n_feats // 3,
                n_feats // 3,
                kernel_size=9,
                stride=1,
                padding=(9 // 2) * 4,
                dilation=4,
                groups=n_feats // 3,
            ),
            nn.Conv3d(n_feats // 3, n_feats // 3, kernel_size=1, stride=1, padding=0),
        )

        self.proj_last = nn.Sequential(nn.Conv3d(n_feats, n_feats, kernel_size=1, stride=1, padding=0))

        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)

        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat(
            [
                self.LKA3(a_1) * self.X3(a_1),
                self.LKA5(a_2) * self.X5(a_2),
                self.LKA7(a_3) * self.X7(a_3),
            ],
            dim=1,
        )

        x = self.proj_last(x * a) * self.scale + shortcut
        return x


__all__ = ["LayerNorm", "MLKA"]
