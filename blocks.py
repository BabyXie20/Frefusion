"""
基本卷积块模块
包含：ConvBlock、ResidualConvBlock、ResidualFuse3D 等基础构建块
"""

import torch
from torch import nn
import torch.nn.functional as F

from .utils import norm3d


# ========== 基础卷积块 ==========
class ConvBlock(nn.Module):
    """
    多阶段卷积块
    
    参数:
        n_stages: 卷积阶段数
        n_filters_in: 输入通道数
        n_filters_out: 输出通道数
        normalization: 归一化方式 ('batchnorm', 'groupnorm', 'instancenorm', 'none')
    
    结构:
        Conv3d(3x3x3) -> Norm -> ReLU
        Conv3d(3x3x3) -> Norm -> ReLU
        ... (重复 n_stages 次)
    
    例子:
        block = ConvBlock(2, 64, 128, normalization='instancenorm')
        out = block(x)  # [B, 128, D, H, W]
    """
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1, bias=False))
            ops.append(norm3d(normalization, n_filters_out))
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DilatedConv3DBlock(nn.Module):
    """
    串联膨胀率为 1、2的 3D 卷积模块，用于提取空间特征。
    通过设置 padding 保持输入/输出的空间尺寸一致。

    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数（默认为与输入相同）
        norm: 归一化方式：'batchnorm' | 'groupnorm' | 'instancenorm' | 'none'
        kernel_size: 卷积核大小（建议使用奇数，例如 3）
        act_layer: 激活函数类（默认为 nn.ReLU）
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 norm: str = 'instancenorm',
                 kernel_size: int = 3,
                 act_layer=nn.ReLU):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        assert kernel_size % 2 == 1, "为了保持尺寸不变，kernel_size 建议为奇数，例如 3。"

        use_bias = (norm == 'none' or norm is None)

        dilations = [1, 2]
        blocks = []
        in_c = in_channels

        for d in dilations:
            padding = d * (kernel_size // 2)  # 确保空间尺寸不变

            conv = nn.Conv3d(
                in_channels=in_c,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=d,
                bias=use_bias
            )

            blocks.append(conv)
            blocks.append(norm3d(norm, out_channels))
            blocks.append(act_layer(inplace=True))

            in_c = out_channels  # 后续层的输入通道等于上一层输出通道

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        """
        x: [N, C, D, H, W]
        返回：同样空间尺寸的特征 [N, out_channels, D, H, W]
        """
        return self.block(x)
    

class ResidualConvBlock(nn.Module):
    """
    多阶段残差卷积块
    
    参数:
        n_stages: 卷积阶段数
        n_filters_in: 输入通道数
        n_filters_out: 输出通道数
        normalization: 归一化方式
    
    结构:
        Conv3d(3x3x3) -> Norm -> ReLU
        Conv3d(3x3x3) -> Norm
        + 残差连接
        ReLU
    
    特点:
        - 支持跳跃连接（残差连接）
        - 最后一个卷积阶段后才应用 ReLU（符合 ResNet 设计）
        - 通道数可变
    
    例子:
        block = ResidualConvBlock(2, 64, 64, normalization='instancenorm')
        out = block(x)  # [B, 64, D, H, W]
    """
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, 3, padding=1, bias=False))
            ops.append(norm3d(normalization, n_filters_out))
            if i != n_stages - 1:  # 最后一个阶段不添加激活，放到外面处理
                ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class ResidualFuse3D(nn.Module):
    """
    3D 残差融合块 - 两层卷积 + 残差连接 + 可选通道投影
    
    参数:
        in_ch: 输入通道数
        out_ch: 输出通道数
        normalization: 归一化方式
    
    结构:
        ├─ Conv3d(3x3x3) -> Norm -> ReLU
        ├─ Conv3d(3x3x3) -> Norm
        ├─ [可选] 1x1x1 投影 (当 in_ch != out_ch 时)
        └─ 残差相加 -> ReLU
    """
    def __init__(self, in_ch, out_ch, normalization='none'):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = norm3d(normalization, out_ch)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = norm3d(normalization, out_ch)

        # 通道投影：当输入输出通道不同时需要投影
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act_out(out)
        return out


class FreqFuse3D(nn.Module):
    """
    频段注意力融合块 - 高低频融合 + 频段/通道注意力
    
    参数:
        channels: 特征通道数
        normalization: 归一化方式
        reduction: 通道注意力的压缩比
    
    输入:
        low: [B, C, D, H, W]      低频子带（LLL）
        highs: [B, C, 7, D, H, W] 高频子带（7个）
    
    输出:
        [B, C, D, H, W] 融合后的特征
    
    工作流程:
        1. 对每个高频子带做 GAP，生成子带特征 [B, 7]
        2. 通过 MLP 生成 band-wise 权重 [B, 7]
        3. 对 7 个高频子带加权融合
        4. 通过 1x1x1 Conv 聚合高频
        5. 与低频相加
        6. 通道注意力加权
        7. 残差细化
    """
    def __init__(self, channels, normalization='instancenorm', reduction=4):
        super().__init__()
        self.channels = channels

        # 7 个高频子带聚合的 1x1x1 卷积
        self.high_agg_conv = nn.Conv3d(channels * 7, channels, kernel_size=1, bias=False)

        # 频段注意力：[B,7] -> [B,7] 的权重
        hidden = max(7 * 2, 8)
        self.band_mlp = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 7),
            nn.Sigmoid()
        )

        # 通道注意力：SE-style
        mid_ch = max(channels // reduction, 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                  # [B, C, 1, 1, 1]
            nn.Conv3d(channels, mid_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )

        # 融合后的局部残差细化
        self.refine = ResidualFuse3D(channels, channels, normalization=normalization)

    def forward(self, low, highs):
        """
        low: [B, C, D, H, W]
        highs: [B, C, 7, D, H, W]
        返回: [B, C, D, H, W]
        """
        B, C, S, D, H, W = highs.shape
        assert S == 7, f"Expect 7 high-frequency subbands, got {S}"

        # 1) 频段描述：全局平均池化
        band_desc = highs.mean(dim=(3, 4, 5))   # [B, C, 7]
        band_desc = band_desc.mean(dim=1)       # [B, 7]

        # 2) MLP 生成 band-wise 权重
        band_weights = self.band_mlp(band_desc)         # [B, 7]
        band_weights = band_weights.view(B, 1, S, 1, 1, 1)  # [B, 1, 7, 1, 1, 1]

        # 3) 对高频子带加权
        highs_weighted = highs * band_weights           # [B, C, 7, D, H, W]

        # 4) 聚合 7 个子带：reshape -> 1x1x1 Conv
        highs_reshaped = highs_weighted.view(B, C * S, D, H, W)  # [B, 7C, D, H, W]
        high_agg = self.high_agg_conv(highs_reshaped)            # [B, C, D, H, W]

        # 5) 与低频融合 + 通道注意力 + 残差细化
        base = low + high_agg                       # [B, C, D, H, W]
        ch_att = self.channel_att(base)             # [B, C, 1, 1, 1]
        fused = base * ch_att                       # [B, C, D, H, W]
        fused = self.refine(fused)                  # [B, C, D, H, W]
        
        return fused


class ALPF3D(nn.Module):
    """
    自适应低通滤波块（3D）- 深度可分离卷积 + 门控

    结构:
        Depthwise Conv3d(3x3x3, groups=C) -> Pointwise Conv3d(1x1x1)
        -> Norm -> Sigmoid 门控
        输出: gate * low_pass + (1 - gate) * x
    """
    def __init__(self, channels, normalization='instancenorm'):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = norm3d(normalization, channels)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        low_pass = self.norm(self.pw(self.dw(x)))
        gate = self.gate(x)
        return low_pass * gate + x * (1 - gate)


class AHPF3D(nn.Module):
    """
    自适应高通增强块（3D）- 深度可分离卷积 + 门控

    结构:
        Depthwise Conv3d -> Pointwise Conv3d -> Norm
        预测门控后强调高频残差: x + gate * (x - filtered)
    """
    def __init__(self, channels, normalization='instancenorm'):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.norm = norm3d(normalization, channels)
        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        filtered = self.norm(self.pw(self.dw(x)))
        gate = self.gate(x)
        return x + gate * (x - filtered)


class Offset3D(nn.Module):
    """
    3D 偏移对齐模块 - 基于 grid_sample 的可学习偏移预测

    输入:
        ref: [B, C, D, H, W] 参考特征（通常为低频）
        feat: [B, C, D, H, W] 需要对齐的特征
    输出:
        aligned_feat: [B, C, D, H, W]
    """
    def __init__(self, channels, normalization='instancenorm'):
        super().__init__()
        in_ch = channels * 2
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv3d(in_ch, channels, kernel_size=1, bias=False)
        self.norm = norm3d(normalization, channels)
        self.gate = nn.Sequential(
            nn.Conv3d(in_ch, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.offset = nn.Conv3d(channels, 3, kernel_size=1, bias=True)

    def forward(self, ref, feat):
        # 深度可分离提取偏移特征
        x = torch.cat([ref, feat], dim=1)
        feat_dw = self.dw(x)
        feat_pw = self.norm(self.pw(feat_dw))
        gated = feat_pw * self.gate(x)

        offset = torch.tanh(self.offset(gated))  # [B, 3, D, H, W]

        B, _, D, H, W = offset.shape
        device = offset.device
        dtype = offset.dtype

        # 构建归一化坐标网格
        zs = torch.linspace(-1.0, 1.0, steps=D, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
        base_grid = torch.stack(torch.meshgrid(zs, ys, xs, indexing='ij'), dim=-1)  # [D, H, W, 3]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, D, H, W, 3]

        offset_grid = offset.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]
        sampling_grid = base_grid + offset_grid

        aligned = F.grid_sample(feat, sampling_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return aligned


# blocks.py 里的一个实现示例
class MultiScaleLocal3D(nn.Module):
    """
    多尺度深度可分离局部特征提取（不使用 DWT 高频）
    输入:  x [B,C,D,H,W]
    输出:  out [B,C,D,H,W]
    """
    def __init__(self, channels, normalization='instancenorm'):
        super().__init__()
        self.channels = channels

        # 1×1×1 Conv（通道混合）
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True)
        )

        # 3×3×3 depthwise + 1×1×1 pointwise
        self.conv3x3 = nn.Sequential(
            nn.Conv3d(
                channels, channels,
                kernel_size=3, padding=1,
                groups=channels, bias=False
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True)
        )

        # 空洞 3×3×3 depthwise + 1×1×1 pointwise
        self.conv3x3_dilate = nn.Sequential(
            nn.Conv3d(
                channels, channels,
                kernel_size=3, padding=2, dilation=2,
                groups=channels, bias=False
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            norm3d(normalization, channels),
            nn.ReLU(inplace=True)
        )

        # 多尺度局部注意力
        self.local_att = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
            norm3d(normalization, channels),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.dropout   = nn.Dropout3d(0.2)

    def forward(self, x):
        # 多尺度局部特征
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv3x3_dilate(x)

        concat_f = torch.cat([f1, f2, f3], dim=1)   # [B,3C,D,H,W]
        local_att = self.local_att(concat_f)        # [B,C,D,H,W]

        fused = (f1 + f2 + f3) * local_att
        out   = self.fuse_conv(fused)
        return self.dropout(out)

class Dilated_Freq_Fuse3D(nn.Module):
    """
    频段注意力融合块 - 高低频融合 + 频段/通道注意力
    
    参数:
        channels: 特征通道数
        normalization: 归一化方式（建议 'instancenorm'）
        reduction: 通道注意力的压缩比
    
    输入:
        low:   [B, C, D, H, W]      低频子带（LLL）
        highs: [B, C, 7, D, H, W]   高频子带（7个）
    
    输出:
        [B, C, D, H, W] 融合后的特征
    
    工作流程:
        0. 低频支路：low -> DilatedConv3DBlock
        1. 对每个高频子带做 GAP，生成子带特征 [B, 7]
        2. 通过 MLP 生成 band-wise 权重 [B, 7]
        3. 对 7 个高频子带加权融合
        4. 通过 1x1x1 Conv 聚合高频
        5. 将 (0) 中的低频特征与 (4) 中的高频特征相加
        6. 通道注意力加权
        7. 残差细化
    """
    def __init__(self, channels, normalization='instancenorm', reduction=4):
        super().__init__()
        self.channels = channels

        # 低频支路：DilatedConv3DBlock（归一化用 InstanceNorm）
        self.low_dilated = DilatedConv3DBlock(
            in_channels=channels,
            out_channels=channels,
            norm=normalization,       # 建议传入 'instancenorm'
            kernel_size=3,
            act_layer=nn.ReLU
        )

        # 7 个高频子带聚合的 1x1x1 卷积
        self.high_agg_conv = nn.Conv3d(
            channels * 7, channels, kernel_size=1, bias=False
        )

        # 频段注意力：[B,7] -> [B,7] 的权重
        hidden = max(7 * 2, 8)
        self.band_mlp = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 7),
            nn.Sigmoid()
        )

        # 通道注意力：SE-style
        mid_ch = max(channels // reduction, 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                  # [B, C, 1, 1, 1]
            nn.Conv3d(channels, mid_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.refine = ResidualFuse3D(
            in_ch=channels,
            out_ch=channels,
            normalization=normalization
        )

    def forward(self, low, highs):
        """
        low:   [B, C, D, H, W]
        highs: [B, C, 7, D, H, W]
        返回:  [B, C, D, H, W]
        """
        B, C, S, D, H, W = highs.shape
        assert S == 7, f"Expect 7 high-frequency subbands, got {S}"

        # 0) 低频支路：DilatedConv3DBlock
        low_dilated = self.low_dilated(low)     # [B, C, D, H, W]

        # 1) 频段描述：全局平均池化
        band_desc = highs.mean(dim=(3, 4, 5))   # [B, C, 7]
        band_desc = band_desc.mean(dim=1)       # [B, 7]

        # 2) MLP 生成 band-wise 权重
        band_weights = self.band_mlp(band_desc)             # [B, 7]
        band_weights = band_weights.view(B, 1, S, 1, 1, 1)  # [B, 1, 7, 1, 1, 1]

        # 3) 对高频子带加权
        highs_weighted = highs * band_weights               # [B, C, 7, D, H, W]

        # 4) 聚合 7 个子带：reshape -> 1x1x1 Conv
        highs_reshaped = highs_weighted.view(
            B, C * S, D, H, W
        )                                                   # [B, 7C, D, H, W]
        high_agg = self.high_agg_conv(highs_reshaped)       # [B, C, D, H, W]

        # 5) 低频(经过 DilatedConv3DBlock) 与高频聚合结果相加
        base = low_dilated + high_agg                       # [B, C, D, H, W]

        # 6) 通道注意力
        ch_att = self.channel_att(base)                     # [B, C, 1, 1, 1]
        fused = base * ch_att                               # [B, C, D, H, W]

        # 7) 残差细化
        fused = self.refine(fused)                          # [B, C, D, H, W]

        return fused


class LH_Filter_Fuse(nn.Module):
    """
    低高频滤波融合模块

    工作流程:
        (a) 可选对 LLL 低频进行 ALPF3D 低通滤波
        (b) 对 7 个高频子带做 band-wise 注意力
        (c) 使用 1x1x1 Conv 聚合高频
        (d) 可选通过 AHPF3D 增强高频
        (e) 可选通过 Offset3D 与低频对齐
        (f) SE 风格通道注意力融合
        (g) ResidualFuse3D 精炼
    """
    def __init__(self, channels, normalization='instancenorm', reduction=4,
                 use_alpf=True, use_ahpf=True, use_offset=True):
        super().__init__()
        self.use_alpf = use_alpf
        self.use_ahpf = use_ahpf
        self.use_offset = use_offset

        self.low_filter = ALPF3D(channels, normalization) if use_alpf else nn.Identity()
        self.high_enhance = AHPF3D(channels, normalization) if use_ahpf else nn.Identity()
        self.offset_align = Offset3D(channels, normalization) if use_offset else nn.Identity()

        self.high_agg_conv = nn.Conv3d(channels * 7, channels, kernel_size=1, bias=False)

        hidden = max(7 * 2, 8)
        self.band_mlp = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 7),
            nn.Sigmoid()
        )

        mid_ch = max(channels // reduction, 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, mid_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.refine = ResidualFuse3D(channels, channels, normalization=normalization)

    def forward(self, low, highs):
        """
        low:   [B, C, D, H, W]
        highs: [B, C, 7, D, H, W]
        返回: [B, C, D, H, W]
        """
        B, C, S, D, H, W = highs.shape
        assert S == 7, f"Expect 7 high-frequency subbands, got {S}"

        low_proc = self.low_filter(low)

        band_desc = highs.mean(dim=(3, 4, 5)).mean(dim=1)  # [B, 7]
        band_weights = self.band_mlp(band_desc).view(B, 1, S, 1, 1, 1)
        highs_weighted = highs * band_weights

        highs_reshaped = highs_weighted.view(B, C * S, D, H, W)
        high_agg = self.high_agg_conv(highs_reshaped)

        high_agg = self.high_enhance(high_agg)

        if self.use_offset:
            aligned_high = self.offset_align(low_proc, high_agg)
        else:
            aligned_high = high_agg

        base = low_proc + aligned_high
        ch_att = self.channel_att(base)
        fused = base * ch_att
        fused = self.refine(fused)
        return fused

# ========== 导出接口 ==========
__all__ = [
    'ConvBlock',
    'ResidualConvBlock',
    'ResidualFuse3D',
    'FreqFuse3D',
    'Dilated_Freq_Fuse3D',
    'MultiScaleLocal3D',
    'ALPF3D',
    'AHPF3D',
    'Offset3D',
    'LH_Filter_Fuse',
]
