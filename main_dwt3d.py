"""
Standalone entry point providing a 3D discrete wavelet transform (DWT) module
implemented purely in PyTorch so it can participate in model training. The
module accepts 5D tensors shaped ``(B, C, D, H, W)`` and returns eight
sub-bands (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH), each shaped
``(B, C, D/2, H/2, W/2)``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class DWT3D(nn.Module):
    """3D Haar Discrete Wavelet Transform implemented with ``torch.conv3d``.

    Notes
    -----
    * The transform is applied independently to every channel in the batch.
    * Spatial dimensions ``D``, ``H`` and ``W`` must be even so they can be
      split into low/high responses.
    * Haar filters are fixed buffers so the module remains differentiable and
      works seamlessly inside training graphs.
    """

    _SUBBAND_MAP: Dict[str, Tuple[str, str, str]] = {
        "LLL": ("L", "L", "L"),
        "LLH": ("L", "L", "H"),
        "LHL": ("L", "H", "L"),
        "LHH": ("L", "H", "H"),
        "HLL": ("H", "L", "L"),
        "HLH": ("H", "L", "H"),
        "HHL": ("H", "H", "L"),
        "HHH": ("H", "H", "H"),
    }

    def __init__(self) -> None:
        super().__init__()
        self.subband_order: Tuple[str, ...] = tuple(self._SUBBAND_MAP.keys())

        scale = 1 / torch.sqrt(torch.tensor(2.0))
        low_1d = torch.tensor([scale, scale])
        high_1d = torch.tensor([scale, -scale])

        kernels = []
        for band in self.subband_order:
            axes = self._SUBBAND_MAP[band]
            depth_filter = low_1d if axes[0] == "L" else high_1d
            height_filter = low_1d if axes[1] == "L" else high_1d
            width_filter = low_1d if axes[2] == "L" else high_1d

            kernel_3d = depth_filter[:, None, None] * height_filter[None, :, None] * width_filter[None, None, :]
            kernels.append(kernel_3d)

        weight = torch.stack(kernels, dim=0).unsqueeze(1)  # (8, 1, 2, 2, 2)
        self.register_buffer("filters", weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute the 3D Haar DWT for a 5D tensor using grouped ``conv3d``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Eight tensors corresponding to the sub-bands ``LLL`` through ``HHH``.
            Each tensor has shape ``(B, C, D/2, H/2, W/2)``.
        """

        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B, C, D, H, W), but got shape {tuple(x.shape)}")

        batch_size, channels, depth, height, width = x.shape
        if depth % 2 or height % 2 or width % 2:
            raise ValueError("D, H, and W dimensions must be divisible by 2 for wavelet decomposition.")

        filters = self.filters.to(dtype=x.dtype)
        weight = filters.repeat(channels, 1, 1, 1, 1)  # (8*C, 1, 2, 2, 2)

        coeffs = F.conv3d(x, weight, stride=2, padding=0, groups=channels)
        coeffs = coeffs.view(batch_size, channels, len(self.subband_order), depth // 2, height // 2, width // 2)

        outputs = []
        for i, name in enumerate(self.subband_order):
            outputs.append(coeffs[:, :, i].contiguous())

        return tuple(outputs)


__all__: Iterable[str] = ("DWT3D",)


if __name__ == "__main__":
    # Minimal sanity check to verify shape propagation without running gradients.
    sample = torch.randn(2, 1, 8, 8, 8)
    transformer = DWT3D()
    bands = transformer(sample)
    print([band.shape for band in bands])
