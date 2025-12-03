"""
医学图像分割工具函数模块
包含：归一化、相似度计算、DWT/IDWT 等通用函数
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


# ========== 归一化工具 ==========
def norm3d(norm, num_channels):
    """
    3D 归一化层工厂函数
    
    参数:
        norm: 'batchnorm', 'groupnorm', 'instancenorm', 'none'
        num_channels: 通道数
    
    返回:
        对应的归一化层
    """
    if norm == 'batchnorm':
        return nn.BatchNorm3d(num_channels)
    elif norm == 'groupnorm':
        return nn.GroupNorm(num_groups=16, num_channels=num_channels)
    elif norm == 'instancenorm':
        return nn.InstanceNorm3d(num_channels)
    elif norm == 'none' or norm is None:
        return nn.Identity()
    else:
        raise ValueError(f'Unknown normalization: {norm}')

class LN(nn.Module):
    def __init__(self, dim):
        super(LN, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x):
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.dim, Ws, Wh, Ww)
        return x



# ========== 相似度计算 ==========
def pairwise_cos_sim(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6):
    """
    计算两组特征的余弦相似度
    
    参数:
        x: [B, N1, C] 或 [N1, C]
        y: [B, N2, C] 或 [N2, C]
        eps: 数值稳定性参数
    
    返回:
        [B, N1, N2] 或 [N1, N2] 的相似度矩阵
        
    例子:
        x = torch.randn(2, 10, 64)  # [batch, seq1, dim]
        y = torch.randn(2, 20, 64)  # [batch, seq2, dim]
        sim = pairwise_cos_sim(x, y)  # [2, 10, 20]
    """
    x_norm = F.normalize(x, dim=-1, eps=eps)
    y_norm = F.normalize(y, dim=-1, eps=eps)
    
    if x.dim() == 3:
        # [B, N1, C] x [B, N2, C] -> [B, N1, N2]
        return torch.einsum('bic,bjc->bij', x_norm, y_norm)
    else:
        # [N1, C] x [N2, C] -> [N1, N2]
        return torch.einsum('ic,jc->ij', x_norm, y_norm)


def pairwise_euclidean_sim(x: torch.Tensor, y: torch.Tensor, temperature: float = 1.0, eps: float = 1e-6):
    """
    x: [B, N1, C] 或 [N1, C]
    y: [B, N2, C] 或 [N2, C]
    返回:
        sim: [B, N1, N2] 或 [N1, N2]，数值越大表示越相似
    """
    dist = torch.cdist(x, y, p=2)  # L2 距离
    # 负的“距离平方”作为打分
    sim = - (dist ** 2) / (temperature ** 2 + eps)
    return sim



# ========== 3D Haar DWT / IDWT ==========
def _haar_pair():
    """生成 Haar 小波的低频和高频滤波器"""
    s2 = math.sqrt(2.0)
    L = torch.tensor([1., 1.]) / s2
    H = torch.tensor([1., -1.]) / s2
    return L, H


def _kron3(a, b, c):
    """
    3D Kronecker 积
    
    参数:
        a, b, c: 1D 张量
    
    返回:
        3D 张量 [i, j, k]
    """
    return torch.einsum('i,j,k->ijk', a, b, c)


def _build_haar_3d_kernels(device):
    """
    构建 3D Haar 小波卷积核
    
    生成 8 个子带的卷积核：
        {LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH}
    
    返回:
        kernels: [8, 1, 2, 2, 2] 
        stride: (2, 2, 2)
        ksize: (2, 2, 2)
    """
    L, H = _haar_pair()
    taps = []
    for zf in [L, H]:
        for yf in [L, H]:
            for xf in [L, H]:
                taps.append(_kron3(zf, yf, xf))  # [2, 2, 2]
    k = torch.stack(taps, dim=0)[:, None, ...]  # [8, 1, 2, 2, 2]
    stride = (2, 2, 2)
    ksize = (2, 2, 2)
    return k.to(device), stride, ksize


class DWT3D(nn.Module):
    """
    3D 离散小波变换 (Discrete Wavelet Transform) - Haar
    输入:  x: [B, C, D, H, W]
    输出:  low:   [B, C, D/2, H/2, W/2]
           highs: [B, C, 7,   D/2, H/2, W/2]
    """
    def __init__(self):
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        # k: [8, 1, 2, 2, 2]
        self.register_buffer("kernels", k)  
        self.stride = stride                 
        self.ksize = ksize                   

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        S = self.kernels.shape[0]           # 8
        # [C*8, 1, 2,2,2]
        weight = self.kernels.repeat(C, 1, 1, 1, 1)

        # groups=C 的分组卷积
        y = F.conv3d(x, weight, stride=self.stride, padding=0, groups=C)
        # y: [B, C*8, D/2, H/2, W/2]

        y = y.view(B, C, S, *y.shape[-3:])  # [B, C, 8, D/2, H/2, W/2]
        low = y[:, :, :1, ...].squeeze(2)   # [B, C, D/2, H/2, W/2]
        highs = y[:, :, 1:, ...]            # [B, C, 7, D/2, H/2, W/2]
        return low, highs


class IDWT3D(nn.Module):
    """
    3D 逆离散小波变换 (Inverse DWT) - Haar
    输入:
        low:   [B, C, D', H', W']
        highs: [B, C, 7, D', H', W']
    输出:
        x: [B, C, 2D', 2H', 2W']
    """
    def __init__(self):
        super().__init__()
        k, stride, ksize = _build_haar_3d_kernels(device=torch.device('cpu'))
        self.register_buffer("kernels", k)   # [8, 1, 2, 2, 2]
        self.stride = stride
        self.ksize = ksize

    def forward(self, low: torch.Tensor, highs: torch.Tensor):
        """
        low:   [B, C, D', H', W']
        highs: [B, C, 7, D', H', W']
        """
        B, C, Dp, Hp, Wp = low.shape
        S = self.kernels.shape[0]           # 8

        y = torch.cat([low.unsqueeze(2), highs], dim=2)  # [B, C, 8, D', H', W']
        y = y.view(B, C * S, Dp, Hp, Wp)                 # [B, C*8, D', H', W']

        weight = self.kernels.repeat(C, 1, 1, 1, 1)      # [C*8, 1, 2, 2, 2]

        x = F.conv_transpose3d(
            y, weight,
            stride=self.stride,
            padding=0,
            output_padding=0,
            groups=C
        )
        return x  # [B, C, 2D', 2H', 2W']



# ========== 导出接口 ==========
__all__ = [
    'norm3d',
    'pairwise_cos_sim',
    'pairwise_euclidean_sim',
    'DWT3D',
    'IDWT3D',
    '_haar_pair',
    '_kron3',
    '_build_haar_3d_kernels',
]
