"""
Standalone entry point providing a 3D discrete wavelet transform (DWT) module
based on ``pywt``. The module accepts 5D tensors shaped ``(B, C, D, H, W)`` and
returns eight sub-bands (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH), each shaped
``(B, C, D/2, H/2, W/2)``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pywt
import torch
from torch import nn


class DWT3D(nn.Module):
    """3D Discrete Wavelet Transform built on ``pywt``.

    Parameters
    ----------
    wavelet : str | pywt.Wavelet
        Wavelet specification forwarded to ``pywt``. Defaults to ``"haar"``.
    mode : str
        Signal extension mode used by ``pywt``. Defaults to ``"reflect"``.

    Notes
    -----
    * The transform is applied independently to every channel in the batch.
    * Spatial dimensions ``D``, ``H`` and ``W`` must be even so they can be
      split into low/high responses.
    """

    _SUBBAND_MAP: Dict[str, str] = {
        "LLL": "aaa",
        "LLH": "aad",
        "LHL": "ada",
        "LHH": "add",
        "HLL": "daa",
        "HLH": "dad",
        "HHL": "dda",
        "HHH": "ddd",
    }

    def __init__(self, wavelet: str | pywt.Wavelet = "haar", mode: str = "reflect") -> None:
        super().__init__()
        self.wavelet = pywt.Wavelet(wavelet) if isinstance(wavelet, str) else wavelet
        self.mode = mode
        self.subband_order: Tuple[str, ...] = tuple(self._SUBBAND_MAP.keys())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute the 3D DWT for a 5D tensor.

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

        device = x.device
        dtype = x.dtype

        # Prepare output containers for all sub-bands.
        outputs = {
            name: torch.empty(
                (batch_size, channels, depth // 2, height // 2, width // 2),
                device=device,
                dtype=dtype,
            )
            for name in self.subband_order
        }

        # Apply the transform per-channel per-batch to leverage ``pywt``'s numpy API.
        for b in range(batch_size):
            for c in range(channels):
                volume = x[b, c].detach().cpu().numpy()
                coeffs = pywt.dwtn(volume, self.wavelet, axes=(0, 1, 2), mode=self.mode)

                for name, pywt_key in self._SUBBAND_MAP.items():
                    band = torch.as_tensor(coeffs[pywt_key], device=device, dtype=dtype)
                    outputs[name][b, c] = band

        return tuple(outputs[name] for name in self.subband_order)


__all__: Iterable[str] = ("DWT3D",)


if __name__ == "__main__":
    # Minimal sanity check to verify shape propagation without running gradients.
    sample = torch.randn(2, 1, 8, 8, 8)
    transformer = DWT3D()
    bands = transformer(sample)
    print([band.shape for band in bands])
