"""
model.py - ECG Classification Models

사용 가능한 모델:
    - baseline           : ECG only (RR feature 미사용)
    - naive_concatenate  : ECG + RR 단순 결합
    - cross_attention    : ECG + RR Cross-Attention

사용법:
    from model import get_model
    model = get_model("baseline", nOUT=4, n_pid=22, **MODEL_CONFIG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# ★ 모델 선택 함수 (이것만 사용하면 됨)
# =============================================================================

def get_model(exp_name: str, nOUT: int, n_pid: int, **config):
    """
    EXP_NAME에 따라 적절한 모델 반환
    
    Args:
        exp_name: 모델 타입이 포함된 이름
                  Improved 시리즈 (Multi-kernel ResNet - 기본): 
                    - baseline_improved, naive_concatenate_improved, cross_attention_improved
                    - baseline, naive_concatenate, cross_attention (동일하게 매핑됨)
                  
        nOUT: 출력 클래스 수 (4: N, S, V, F)
        n_pid: 환자 수 (현재 미사용, 인터페이스 유지용)
        **config: MODEL_CONFIG (in_channels, out_ch, mid_ch, num_heads, n_rr)
    
    Returns:
        model: 해당 모델 인스턴스
    """
    # Improved 시리즈 (Multi-kernel Residual Blocks) - 기본 모델
    models_improved = {
        "baseline": ImprovedDeepResidualCNN_baseline,
        "naive_concatenate": ImprovedDeepResidualCNN_naive_concatenate,
        "cross_attention": ImprovedDeepResidualCNN_CrossAttention,
        "baseline_improved": ImprovedDeepResidualCNN_baseline,
        "naive_concatenate_improved": ImprovedDeepResidualCNN_naive_concatenate,
        "cross_attention_improved": ImprovedDeepResidualCNN_CrossAttention,
    }
    
    # exp_name에서 모델 타입 추출
    model_type = None
    for mt in models_improved.keys():
        if mt in exp_name:
            model_type = mt
            break
    
    if model_type is None:
        raise ValueError(f"Unknown model in exp_name: '{exp_name}'. "
                         f"Must contain one of: {list(models_improved.keys())}")
    
    model_class = models_improved[model_type]
    model = model_class(nOUT=nOUT, n_pid=n_pid, **config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {model_class.__name__}")
    print(f"  - Exp   : {exp_name}")
    print(f"  - Type  : {model_type}")
    print(f"  - Params: {n_params:,}")
    
    return model


# =============================================================================
# Building Blocks
# =============================================================================

class Bottleneck(nn.Module):
    """DenseNet Bottleneck block"""
    def __init__(self, nChannels: int, growthRate: int):
        super().__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(interChannels)
        self.conv2 = nn.Conv1d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat((x, out), 1)


class Transition(nn.Module):
    """DenseNet Transition block"""
    def __init__(self, nChannels: int, nOutChannels: int, down: bool = False):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.down = down

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.down:
            out = F.avg_pool1d(out, 2)
        return out


class ResidualUBlock(nn.Module):
    """U-Net style Residual Block for 1D signals"""
    def __init__(self, out_ch: int, mid_ch: int, layers: int, downsampling: bool = True):
        super().__init__()
        self.downsample = downsampling
        K, P = 9, 4

        self.conv1 = nn.Conv1d(out_ch, out_ch, K, padding=P, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for idx in range(layers):
            in_ch = out_ch if idx == 0 else mid_ch
            self.encoders.append(nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, K, stride=2, padding=P, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.LeakyReLU()
            ))

            out_ch_dec = out_ch if idx == layers - 1 else mid_ch
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose1d(mid_ch * 2, out_ch_dec, K, stride=2, padding=P, output_padding=1, bias=False),
                nn.BatchNorm1d(out_ch_dec),
                nn.LeakyReLU()
            ))

        self.bottleneck = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, K, padding=P, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU()
        )

        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(2, 2)
            self.idfunc_1 = nn.Conv1d(out_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))

        encoder_out = []
        out = x_in
        for layer in self.encoders:
            out = layer(out)
            encoder_out.append(out)

        out = self.bottleneck(out)

        for idx, layer in enumerate(self.decoders):
            skip = encoder_out[-1 - idx]
            if out.size(-1) != skip.size(-1):
                out = F.interpolate(out, size=skip.size(-1), mode='linear', align_corners=False)
            out = layer(torch.cat([out, skip], dim=1))

        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]

        out += x_in

        if self.downsample:
            out = self.idfunc_1(self.idfunc_0(out))

        return out


# =============================================================================
# DEPRECATED: Old Models (A & B Series) - Replaced with Improved Architecture
# =============================================================================
# 이전 모델들 (ResU_Dense_*, ResU_*_NoDense)은 새로운 
# ImprovedDeepResidualCNN 모델로 통합되었습니다.
# 
# 기본 모델은 다음과 같이 매핑됩니다:
#  - "baseline" → ImprovedDeepResidualCNN_baseline
#  - "naive_concatenate" → ImprovedDeepResidualCNN_naive_concatenate
#  - "cross_attention" → ImprovedDeepResidualCNN_CrossAttention


# =============================================================================
# Improved Deep Residual CNN (Paper Architecture)
# Multi-kernel Residual Blocks with Cross-Attention
# =============================================================================

class MultiKernelConvBlock(nn.Module):
    """
    Convolution block with multiple kernel sizes (improved perception of different scales)
    Kernel sizes: [base_kernel_size, base_kernel_size+8, base_kernel_size+14]
    Each with num_kernels filters
    """
    def __init__(self, in_channels: int, num_kernels: int, base_kernel_size: int = 28):
        super().__init__()
        kernel_sizes = [base_kernel_size, base_kernel_size + 8, base_kernel_size + 14]
        
        self.convs = nn.ModuleList()
        for k_size in kernel_sizes:
            self.convs.append(
                nn.Conv1d(in_channels, num_kernels, kernel_size=k_size, 
                         padding=k_size//2, bias=False)
            )
        self.bn = nn.BatchNorm1d(num_kernels * len(kernel_sizes))
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ImprovedResidualBlock(nn.Module):
    """
    Improved Residual Block with multi-kernel convolutions
    Each block contains 2 conv layers with different kernel combinations
    """
    def __init__(self, in_channels: int, base_kernel_size_1: int, 
                 base_kernel_size_2: int, num_kernels: int, downsample: bool = False):
        super().__init__()
        
        self.downsample_flag = downsample
        
        # First multi-kernel conv layer
        self.conv_block1 = MultiKernelConvBlock(in_channels, num_kernels, base_kernel_size_1)
        out_ch_1 = num_kernels * 3  # 3 kernel sizes
        
        # Dropout
        self.dropout1 = nn.Dropout(0.5)
        
        # Second multi-kernel conv layer
        self.conv_block2 = MultiKernelConvBlock(out_ch_1, num_kernels, base_kernel_size_2)
        out_ch_2 = num_kernels * 3  # 3 kernel sizes
        
        # Dropout
        self.dropout2 = nn.Dropout(0.5)
        
        # 1x1 conv to match dimensions for residual connection
        self.conv_1x1 = nn.Conv1d(in_channels, out_ch_2, kernel_size=1, bias=False)
        self.bn_1x1 = nn.BatchNorm1d(out_ch_2)
        
        # Downsampling (Max Pooling)
        if downsample:
            self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Save input for residual
        identity = x
        
        # First conv block
        out = self.conv_block1(x)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv_block2(out)
        out = self.dropout2(out)
        
        # Residual connection (adjust identity dimensions if needed)
        identity = self.conv_1x1(identity)
        identity = self.bn_1x1(identity)
        
        out = out + identity
        
        # Downsampling
        if self.downsample_flag:
            out = self.maxpool(out)
        
        return out


class ImprovedDeepResidualCNN_baseline(nn.Module):
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=48,
                 n_rr=7,
                 fusion_type="none",
                 **fusion_kwargs):
        super().__init__()

        self.initial_conv = MultiKernelConvBlock(in_channels, out_ch // 3, base_kernel_size=28)

        self.residual_blocks = nn.ModuleList()
        kernel_configs = [(32,34),(32,34),(30,36),(30,36),(30,36),(30,36),(30,36),(30,36)]

        in_ch = out_ch * 3
        for i,(k1,k2) in enumerate(kernel_configs):
            downsample = (i % 2 == 1)
            self.residual_blocks.append(
                ImprovedResidualBlock(in_ch, k1, k2, out_ch, downsample=downsample)
            )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # ✅ No fusion
        self.fusion = None
        fusion_out_dim = out_ch * 3

        self.fc = nn.Sequential(
            nn.Linear(fusion_out_dim, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features=None, rr_remove_ablation=False):

        x = self.initial_conv(ecg_signal)
        for block in self.residual_blocks:
            x = block(x)

        z = self.avgpool(x).squeeze(-1)  # (B,C)

        logits = self.fc(z)
        return logits, None


class ImprovedDeepResidualCNN_naive_concatenate(nn.Module):
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=48,
                 n_rr=7,
                 fusion_type="concat",
                 **fusion_kwargs):
        super().__init__()

        self.initial_conv = MultiKernelConvBlock(in_channels, out_ch // 3, base_kernel_size=28)

        self.residual_blocks = nn.ModuleList()
        kernel_configs = [(32,34),(32,34),(30,36),(30,36),(30,36),(30,36),(30,36),(30,36)]

        in_ch = out_ch * 3
        for i,(k1,k2) in enumerate(kernel_configs):
            downsample = (i % 2 == 1)
            self.residual_blocks.append(
                ImprovedResidualBlock(in_ch, k1, k2, out_ch, downsample=downsample)
            )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # ✅ Fusion block (Concat)
        self.fusion = get_fusion_block(
            fusion_type,
            emb_dim=out_ch * 3,
            rr_dim=n_rr,
            **fusion_kwargs
        )

        fusion_out_dim = out_ch * 3 + n_rr

        self.fc = nn.Sequential(
            nn.Linear(fusion_out_dim, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):

        x = self.initial_conv(ecg_signal)
        for block in self.residual_blocks:
            x = block(x)

        z = self.avgpool(x).squeeze(-1)

        if rr_remove_ablation:
            rr_features = torch.zeros_like(rr_features)

        # ✅ Fusion 적용
        fused = self.fusion(z, rr_features)

        logits = self.fc(fused)
        return logits, None


class ImprovedDeepResidualCNN_CrossAttention(nn.Module):
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=48,
                 n_rr=7,
                 num_heads=1,
                 fusion_type="mhca",
                 **fusion_kwargs):
        super().__init__()

        self.initial_conv = MultiKernelConvBlock(in_channels, out_ch // 3, base_kernel_size=28)

        self.residual_blocks = nn.ModuleList()
        kernel_configs = [(32,34),(32,34),(30,36),(30,36),(30,36),(30,36),(30,36),(30,36)]

        in_ch = out_ch * 3
        for i,(k1,k2) in enumerate(kernel_configs):
            downsample = (i % 2 == 1)
            self.residual_blocks.append(
                ImprovedResidualBlock(in_ch, k1, k2, out_ch, downsample=downsample)
            )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # ✅ Fusion block (Gated MHCA)
        self.fusion = get_fusion_block(
            fusion_type,
            emb_dim=out_ch * 3,
            rr_dim=n_rr,
            num_heads=num_heads,
            **fusion_kwargs
        )

        fusion_out_dim = 2 * (out_ch * 3)

        self.fc = nn.Sequential(
            nn.Linear(fusion_out_dim, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):

        x = self.initial_conv(ecg_signal)
        for block in self.residual_blocks:
            x = block(x)

        z = self.avgpool(x).squeeze(-1)

        if rr_remove_ablation:
            rr_features = torch.zeros_like(rr_features)

        # ✅ Fusion 적용
        fused = self.fusion(z, rr_features)

        logits = self.fc(fused)

        return logits, None
