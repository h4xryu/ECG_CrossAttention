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
                  A 시리즈 (Dense Block 포함): baseline_A, naive_concatenate_A, cross_attention_A
                  B 시리즈 (Dense Block 없음): baseline_B, naive_concatenate_B, cross_attention_B
        nOUT: 출력 클래스 수 (4: N, S, V, F)
        n_pid: 환자 수 (현재 미사용, 인터페이스 유지용)
        **config: MODEL_CONFIG (in_channels, out_ch, mid_ch, num_heads, n_rr)
    
    Returns:
        model: 해당 모델 인스턴스
    """
    # B 시리즈 (Dense Block 없음) 먼저 체크
    models_b = {
        "baseline_B": ResU_baseline_NoDense,
        "naive_concatenate_B": ResU_naive_concatenate_NoDense,
        "cross_attention_B": ResU_CrossAttention_NoDense,
    }
    
    # A 시리즈 (Dense Block 포함) - 기본값
    models_a = {
        "baseline": ResU_Dense_baseline,
        "naive_concatenate": ResU_Dense_naive_concatenate,
        "cross_attention": ResU_Dense_CrossAttention,
    }
    
    # 전체 모델 딕셔너리
    models = {**models_b, **models_a}
    
    # exp_name에서 모델 타입 추출 (baseline-xxx -> baseline)
    model_type = None
    for mt in models.keys():
        if mt in exp_name:
            model_type = mt
            break
    
    if model_type is None:
        raise ValueError(f"Unknown model in exp_name: '{exp_name}'. "
                         f"Must contain one of: {list(models.keys())}")
    
    model_class = models[model_type]
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
# Model 1: Baseline (ECG only, RR 미사용)
# =============================================================================

class ResU_Dense_baseline(nn.Module):
    """
    Baseline model - ECG 신호만 사용, RR feature 미사용
    Self-Attention으로 ECG temporal pattern 학습
    """
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=9):
        super().__init__()
        
        # ECG Encoder
        self.conv = nn.Conv1d(in_channels, out_ch, 15, padding=7, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=4)
        self.rub_1 = ResidualUBlock(out_ch, mid_ch, layers=3)
        
        # Self-Attention
        self.mha_ecg = nn.MultiheadAttention(out_ch, num_heads, dropout=0.2)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(out_ch, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        # ECG encoding
        x = F.leaky_relu(self.bn(self.conv(ecg_signal)))
        x = self.rub_0(x)
        x = self.rub_1(x)
        x_seq = x.permute(2, 0, 1)  # (L, B, E)
        
        # Self-attention (RR 미사용)
        z, attn = self.mha_ecg(x_seq, x_seq, x_seq)
        z = z.mean(dim=0)  # (B, E)
        
        logits = self.fc(z)
        return logits, attn


# =============================================================================
# Model 2: Naive Concatenate (ECG + RR 단순 결합)
# =============================================================================

class ResU_Dense_naive_concatenate(nn.Module):
    """
    ECG Self-Attention 결과와 RR feature를 단순 concatenate
    """
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=9):
        super().__init__()
        
        # ECG Encoder
        self.conv = nn.Conv1d(in_channels, out_ch, 15, padding=7, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=4)
        self.rub_1 = ResidualUBlock(out_ch, mid_ch, layers=3)
        
        # Self-Attention
        self.mha_ecg = nn.MultiheadAttention(out_ch, num_heads, dropout=0.2)
        
        # Classifier (ECG + RR)
        self.fc = nn.Sequential(
            nn.Linear(out_ch+n_rr, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        # ECG encoding
        x = F.leaky_relu(self.bn(self.conv(ecg_signal)))
        x = self.rub_0(x)
        x = self.rub_1(x)
        x_seq = x.permute(2, 0, 1)  # (L, B, E)
        
        # Self-attention
        z, attn = self.mha_ecg(x_seq, x_seq, x_seq)
        z = z.mean(dim=0)  # (B, E)
        
        # RR concatenate
        logits = self.fc(torch.cat([z, rr_features], dim=1))
        return logits, attn


# =============================================================================
# Model 3: Cross-Attention (RR → ECG attention)
# =============================================================================

class ResU_Dense_CrossAttention(nn.Module):
    """
    RR feature를 Query로 사용하여 ECG sequence에 Cross-Attention
    RR 정보가 ECG의 어느 부분에 주목할지 학습
    """
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=9):
        super().__init__()
        
        # ECG Encoder
        self.conv = nn.Conv1d(in_channels, out_ch, 15, padding=7, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=4)
        self.rub_1 = ResidualUBlock(out_ch, mid_ch, layers=3)
        
        # RR Encoder (RR -> Query embedding)
        self.rr_encoder = nn.Linear(n_rr, out_ch)
        
        # Cross-Attention (Query: RR, Key/Value: ECG)
        self.mha_ecg = nn.MultiheadAttention(out_ch, num_heads, dropout=0.2)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(out_ch, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        # ECG encoding
        x = F.leaky_relu(self.bn(self.conv(ecg_signal)))
        x = self.rub_0(x)
        x = self.rub_1(x)
        x_seq = x.permute(2, 0, 1)  # (L, B, E)
        
        # RR encoding -> Query
        rr_emb = self.rr_encoder(rr_features).unsqueeze(0)  # (1, B, E)
        if rr_remove_ablation:
            rr_emb = torch.zeros_like(rr_emb)
        
        # Cross-attention (RR attends to ECG)
        z, attn = self.mha_ecg(rr_emb, x_seq, x_seq)
        z = z.squeeze(0)  # (B, E)
        
        # RR concatenate for residual connection
        logits = self.fc(z)
        return logits, attn


# =============================================================================
# B Series: Models WITHOUT Dense Block (simpler architecture)
# =============================================================================

class ResU_baseline_NoDense(nn.Module):
    """
    B0: Baseline model without Dense Block - ECG only
    ResU Block -> AvgPool -> Linear
    """
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=9):
        super().__init__()
        
        # ECG Encoder (no Dense Block)
        self.conv = nn.Conv1d(in_channels, out_ch, 15, padding=7, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=4)
        self.rub_1 = ResidualUBlock(out_ch, mid_ch, layers=3, downsampling=False)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(out_ch, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        # ECG encoding
        x = F.leaky_relu(self.bn(self.conv(ecg_signal)))
        x = self.rub_0(x)
        x = self.rub_1(x)
        
        # Global average pooling
        z = self.avgpool(x).squeeze(-1)  # (B, C)
        
        logits = self.fc(z)
        return logits, None


class ResU_naive_concatenate_NoDense(nn.Module):
    """
    B1: Naive Concatenate without Dense Block
    ECG AvgPool + RR concat -> Linear
    """
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=9):
        super().__init__()
        
        # ECG Encoder (no Dense Block)
        self.conv = nn.Conv1d(in_channels, out_ch, 15, padding=7, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=4)
        self.rub_1 = ResidualUBlock(out_ch, mid_ch, layers=3, downsampling=False)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier (ECG + RR)
        self.fc = nn.Sequential(
            nn.Linear(out_ch + n_rr, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        # ECG encoding
        x = F.leaky_relu(self.bn(self.conv(ecg_signal)))
        x = self.rub_0(x)
        x = self.rub_1(x)
        
        # Global average pooling
        z = self.avgpool(x).squeeze(-1)  # (B, C)
        
        # RR concatenate
        if rr_remove_ablation:
            rr_features = torch.zeros_like(rr_features)
        
        logits = self.fc(torch.cat([z, rr_features], dim=1))
        return logits, None


class ResU_CrossAttention_NoDense(nn.Module):
    """
    B2: Cross-Attention without Dense Block
    RR Query attends to ECG sequence -> AvgPool -> Linear
    """
    def __init__(self, nOUT, n_pid,
                 in_channels=1, out_ch=180, mid_ch=30, n_rr=7, num_heads=9):
        super().__init__()
        
        # ECG Encoder (no Dense Block)
        self.conv = nn.Conv1d(in_channels, out_ch, 15, padding=7, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=4)
        self.rub_1 = ResidualUBlock(out_ch, mid_ch, layers=3, downsampling=False)
        
        # RR Encoder
        self.rr_encoder = nn.Linear(n_rr, out_ch)
        
        # Cross-Attention (Query: RR, Key/Value: ECG)
        self.mha_cross = nn.MultiheadAttention(out_ch, num_heads, dropout=0.2)
        
        # Post-attention processing
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(out_ch, 360),
            nn.LayerNorm(360),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(360, nOUT)
        )

    def forward(self, ecg_signal, rr_features, rr_remove_ablation=False):
        # ECG encoding
        x = F.leaky_relu(self.bn(self.conv(ecg_signal)))
        x = self.rub_0(x)
        x = self.rub_1(x)
        x_seq = x.permute(2, 0, 1)  # (L, B, C)
        
        # RR encoding -> Query
        rr_emb = self.rr_encoder(rr_features).unsqueeze(0)  # (1, B, C)
        if rr_remove_ablation:
            rr_emb = torch.zeros_like(rr_emb)
        
        # Cross-attention
        z, attn = self.mha_cross(rr_emb, x_seq, x_seq)
        z = z.squeeze(0)  # (B, C)
        
        logits = self.fc(z)
        return logits, attn
