import os
import random
import hashlib
import json
import numpy as np
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt
from tqdm import tqdm
import torch.nn as nn
from config import CLASSES, LABEL_TO_ID, LABEL_GROUP_MAP, RR_FEATURE_OPTION
from joblib import Parallel, delayed
import multiprocessing as mp

# =============================================================================
# Dataset Cache Directory
# =============================================================================
CACHE_DIR = './dataset'

# =============================================================================
# Seed Setting
# =============================================================================

def set_seed(seed: int, fully_deterministic: bool = True) -> None:
    """
    랜덤 시드 설정
    
    Args:
        seed: 랜덤 시드
        fully_deterministic: True면 완전 결정론적 (느림), False면 기본 설정
    """
    import os
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 기본 cuDNN 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    



# =============================================================================
# Signal Processing
# =============================================================================

def resample_to_len(ts: np.ndarray, out_len: int) -> np.ndarray:
    if len(ts) == out_len:
        return ts.astype(np.float32)
    if len(ts) < 2:
        return np.pad(ts.astype(np.float32), (0, max(0, out_len - len(ts))), mode='edge')[:out_len]

    x_old = np.linspace(0.0, 1.0, num=len(ts))
    x_new = np.linspace(0.0, 1.0, num=out_len)
    return interp1d(x_old, ts, kind='linear')(x_new).astype(np.float32)


def remove_baseline_bandpass(signal: np.ndarray, fs: int = 360, 
                              lowcut: float = 0.1, highcut: float = 100.0, 
                              order: int = 256) -> np.ndarray:
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    fir_coeff = firwin(order + 1, [low, high], pass_zero=False)
    return filtfilt(fir_coeff, 1.0, signal)


# =============================================================================
# RR Interval + Morphological Features
# =============================================================================

def get_rr_feature_function(option: str):
    """
    RR feature 옵션에 따라 적절한 함수 반환
    
    Args:
        option: "opt1", "opt2", "opt3", "opt4"
    
    Returns:
        feature extraction function
    """
    functions = {
        "opt1": compute_rr_features_opt1,  # 7 features (basic)
        "opt2": compute_rr_features_opt2,  # 38 features (full + morphology)
        "opt3": compute_rr_features_opt3,  # 7 features (basic2)
        "opt4": compute_rr_features_opt4,  # 7 features (numerically stable)
    }
    
    if option not in functions:
        raise ValueError(f"Unknown RR option: '{option}'. Available: {list(functions.keys())}")
    
    return functions[option]


def compute_rr_features_opt1(
    ecg_signal: np.ndarray, 
    r_peaks: np.ndarray, 
    fs: int = 360
) -> np.ndarray:
    """
    Basic RR interval features (7 features)
    
    Features:
        [0] pre_rr        - Current RR interval (ms)
        [1] post_rr       - Next RR interval (ms)
        [2] local_rr      - Local mean RR (last 10 beats)
        [3] local_std     - Local RR std
        [4] local_rmssd   - Local RMSSD
        [5] rr_pre_div_cur- RR_{i-2} / RR_i
        [6] pre_div_post  - RR_i / RR_{i+1}
    
    Returns:
        features: (n_beats, 7) array
    """
    n_beats = len(r_peaks)
    
    # Convert to ms units
    ms_factor = 1000.0 / fs
    
    # =========================================================================
    # Part 1: RR Interval Features (original 26 features)
    # =========================================================================
    
    # Initialize RR arrays
    pre_rr = np.zeros(n_beats, dtype=np.float32)
    post_rr = np.zeros(n_beats, dtype=np.float32)
    local_rr = np.zeros(n_beats, dtype=np.float32)
    local_rmssd = np.zeros(n_beats, dtype=np.float32)
    local_std = np.zeros(n_beats, dtype=np.float32)
    global_skew = np.zeros(n_beats, dtype=np.float32)
    global_kurt = np.zeros(n_beats, dtype=np.float32)
    
    rr_cur_div_avg = np.zeros(n_beats, dtype=np.float32)
    rr_post_div_cur = np.zeros(n_beats, dtype=np.float32)
    rr_pre_div_cur = np.zeros(n_beats, dtype=np.float32)
    rr_post_div_avg = np.zeros(n_beats, dtype=np.float32)
    
    tr_half = np.zeros(n_beats, dtype=np.float32)
    tr_quarter = np.zeros(n_beats, dtype=np.float32)
    tr_half_norm = np.zeros(n_beats, dtype=np.float32)
    tr_quarter_norm = np.zeros(n_beats, dtype=np.float32)
    
    rr_global_normalized = np.zeros(n_beats, dtype=np.float32)
    rr1_squared = np.zeros(n_beats, dtype=np.float32)
    rr2_exp = np.zeros(n_beats, dtype=np.float32)
    rr3_exp_inv = np.zeros(n_beats, dtype=np.float32)
    rr4_log = np.zeros(n_beats, dtype=np.float32)
    
    # Pre RR and Post RR
    for i in range(n_beats):
        if i > 0:
            pre_rr[i] = (r_peaks[i] - r_peaks[i-1]) * ms_factor
        else:
            if n_beats > 1:
                pre_rr[i] = (r_peaks[1] - r_peaks[0]) * ms_factor
            else:
                pre_rr[i] = 800.0
        
        if i < n_beats - 1:
            post_rr[i] = (r_peaks[i+1] - r_peaks[i]) * ms_factor
        else:
            if n_beats > 1:
                post_rr[i] = (r_peaks[-1] - r_peaks[-2]) * ms_factor
            else:
                post_rr[i] = 800.0
    
    # Local statistics
    for i in range(n_beats):
        start_idx = max(0, i - 9)
        window = pre_rr[start_idx:i+1]
        if len(window) > 0:
            local_rr[i] = np.mean(window)
            local_std[i] = np.std(window) if len(window) > 1 else 0.0
        else:
            local_rr[i] = pre_rr[i]
            local_std[i] = 0.0
    
    # Local RMSSD
    for i in range(n_beats):
        start = max(0, i - 9)
        rr_window = pre_rr[start:i+1]
        
        if len(rr_window) >= 2:
            diff_rr = np.diff(rr_window)
            local_rmssd[i] = np.sqrt(np.mean(diff_rr ** 2))
        else:
            local_rmssd[i] = 0.0
    
    # Global statistics
    valid_pre_rr = pre_rr[pre_rr > 50]
    
    if len(valid_pre_rr) > 1:
        global_rr_mean = np.mean(valid_pre_rr)
        global_rr_std = np.std(valid_pre_rr)
        global_rmssd_value = np.sqrt(np.mean(np.diff(valid_pre_rr) ** 2))
        
        if len(valid_pre_rr) >= 3:
            from scipy.stats import skew, kurtosis
            global_skew_value = float(skew(valid_pre_rr))
            global_kurt_value = float(kurtosis(valid_pre_rr)) if len(valid_pre_rr) >= 4 else 0.0
        else:
            global_skew_value = 0.0
            global_kurt_value = 0.0
    else:
        global_rr_mean = 800.0
        global_rr_std = 50.0
        global_rmssd_value = 50.0
        global_skew_value = 0.0
        global_kurt_value = 0.0
    
    global_rr = np.full(n_beats, global_rr_mean, dtype=np.float32)
    global_rr_std_arr = np.full(n_beats, global_rr_std, dtype=np.float32)
    global_rmssd = np.full(n_beats, global_rmssd_value, dtype=np.float32)
    global_skew[:] = global_skew_value
    global_kurt[:] = global_kurt_value
    
    # Derived features
    epsilon = 1.0
    
    pre_minus_global = pre_rr - global_rr
    pre_div_global = pre_rr / np.maximum(global_rr, epsilon)
    pre_div_post = pre_rr / np.maximum(post_rr, epsilon)
    
    # RR ratio features
    rr_cur_div_avg = pre_rr / np.maximum(global_rr, epsilon)
    rr_post_div_cur = post_rr / np.maximum(pre_rr, epsilon)
    
    for i in range(n_beats):
        if i > 0:
            prev_prev_rr = (r_peaks[i-1] - r_peaks[i-2]) * ms_factor if i > 1 else pre_rr[i]
            rr_pre_div_cur[i] = prev_prev_rr / max(pre_rr[i], epsilon)
        else:
            rr_pre_div_cur[i] = 1.0
    
    rr_post_div_avg = post_rr / np.maximum(global_rr, epsilon)
    
    # Temporal features
    for i in range(n_beats):
        if i < n_beats - 1:
            tr_half[i] = post_rr[i] / 2.0
            tr_quarter[i] = post_rr[i] / 4.0
            
            tr_half_norm[i] = tr_half[i] / max(global_rr_mean, epsilon)
            tr_quarter_norm[i] = tr_quarter[i] / max(global_rr_mean, epsilon)
        else:
            tr_half[i] = post_rr[i] / 2.0
            tr_quarter[i] = post_rr[i] / 4.0
            tr_half_norm[i] = tr_half[i] / max(global_rr_mean, epsilon)
            tr_quarter_norm[i] = tr_quarter[i] / max(global_rr_mean, epsilon)
    
    # RR_global transformations
    rr_global_normalized = pre_rr / np.maximum(global_rr, epsilon)
    rr_global_clipped = np.clip(rr_global_normalized, 0.01, 100.0)
    
    rr1_squared = rr_global_clipped ** 2
    rr2_exp = np.exp(np.clip(rr_global_clipped, -10, 10))
    rr3_exp_inv = np.exp(np.clip(1.0 / rr_global_clipped, -10, 10))
    rr4_log = np.log(rr_global_clipped)
    
    # =========================================================================
    # Part 2: Morphological Features (PR interval, QRS width) - NEW
    # =========================================================================
    
    # Initialize morphological arrays
    pr_interval = np.zeros(n_beats, dtype=np.float32)
    qrs_width = np.zeros(n_beats, dtype=np.float32)
    
    local_pr_mean = np.zeros(n_beats, dtype=np.float32)
    local_pr_std = np.zeros(n_beats, dtype=np.float32)
    local_qrs_mean = np.zeros(n_beats, dtype=np.float32)
    local_qrs_std = np.zeros(n_beats, dtype=np.float32)
    
    global_pr_mean = np.zeros(n_beats, dtype=np.float32)
    global_pr_std = np.zeros(n_beats, dtype=np.float32)
    global_qrs_mean = np.zeros(n_beats, dtype=np.float32)
    global_qrs_std = np.zeros(n_beats, dtype=np.float32)
    
    pr_normalized = np.zeros(n_beats, dtype=np.float32)
    qrs_normalized = np.zeros(n_beats, dtype=np.float32)
    
    try:
    #     # CRITICAL FIX: Ensure r_peaks is integer array and properly formatted
    #     r_peaks_int = np.asarray(r_peaks, dtype=np.int64).flatten()
        
    #     # Validate signal length
    #     if len(ecg_signal) < 100:
    #         raise ValueError("ECG signal too short for delineation")
        
    #     # Detect P peaks, Q peaks, S peaks using NeuroKit2
    #     _, waves_peak = nk.ecg_delineate(
    #         ecg_signal, 
    #         rpeaks=r_peaks_int,  # Use 'rpeaks' parameter name
    #         sampling_rate=fs,
    #         method="dwt"
    #     )
        
    #     # Extract peak arrays - handle both dict and DataFrame returns
    #     if isinstance(waves_peak, dict):
    #         p_peaks = waves_peak.get('ECG_P_Peaks', None)
    #         q_peaks = waves_peak.get('ECG_Q_Peaks', None)
    #         s_peaks = waves_peak.get('ECG_S_Peaks', None)
    #     else:
    #         # If it's a DataFrame
    #         p_peaks = waves_peak['ECG_P_Peaks'].values if 'ECG_P_Peaks' in waves_peak.columns else None
    #         q_peaks = waves_peak['ECG_Q_Peaks'].values if 'ECG_Q_Peaks' in waves_peak.columns else None
    #         s_peaks = waves_peak['ECG_S_Peaks'].values if 'ECG_S_Peaks' in waves_peak.columns else None
        
    #     # Filter out NaN values
    #     if p_peaks is not None:
    #         p_peaks = p_peaks[~np.isnan(p_peaks)].astype(np.int64)
    #     if q_peaks is not None:
    #         q_peaks = q_peaks[~np.isnan(q_peaks)].astype(np.int64)
    #     if s_peaks is not None:
    #         s_peaks = s_peaks[~np.isnan(s_peaks)].astype(np.int64)
        
    #     # ---------------------------------------------------------------------
    #     # Compute PR interval and QRS width for each beat
    #     # ---------------------------------------------------------------------
    #     for i in range(n_beats):
    #         r_idx = r_peaks_int[i]
            
    #         # PR interval: P peak to R peak (ms)
    #         if p_peaks is not None and len(p_peaks) > 0:
    #             valid_p = p_peaks[p_peaks < r_idx]
    #             if len(valid_p) > 0:
    #                 p_idx = valid_p[-1]
    #                 pr_interval[i] = (r_idx - p_idx) * ms_factor
    #             else:
    #                 pr_interval[i] = 160.0
    #         else:
    #             pr_interval[i] = 160.0
            
    #         # QRS width: Q peak to S peak (ms)
    #         if q_peaks is not None and s_peaks is not None and len(q_peaks) > 0 and len(s_peaks) > 0:
    #             valid_q = q_peaks[q_peaks <= r_idx]
    #             valid_s = s_peaks[s_peaks >= r_idx]
                
    #             if len(valid_q) > 0 and len(valid_s) > 0:
    #                 q_idx = valid_q[-1]
    #                 s_idx = valid_s[0]
    #                 qrs_width[i] = (s_idx - q_idx) * ms_factor
    #             else:
    #                 qrs_width[i] = 100.0
    #         else:
    #             qrs_width[i] = 100.0
        
        # ---------------------------------------------------------------------
        # Local statistics (last 10 beats window)
        # ---------------------------------------------------------------------
        for i in range(n_beats):
            start_idx = max(0, i - 9)
            
            # PR local stats
            pr_window = pr_interval[start_idx:i+1]
            if len(pr_window) > 0:
                local_pr_mean[i] = np.mean(pr_window)
                local_pr_std[i] = np.std(pr_window) if len(pr_window) > 1 else 0.0
            else:
                local_pr_mean[i] = pr_interval[i]
                local_pr_std[i] = 0.0
            
            # QRS local stats
            qrs_window = qrs_width[start_idx:i+1]
            if len(qrs_window) > 0:
                local_qrs_mean[i] = np.mean(qrs_window)
                local_qrs_std[i] = np.std(qrs_window) if len(qrs_window) > 1 else 0.0
            else:
                local_qrs_mean[i] = qrs_width[i]
                local_qrs_std[i] = 0.0
        
        # ---------------------------------------------------------------------
        # Global statistics
        # ---------------------------------------------------------------------
        # valid_pr = pr_interval[(pr_interval > 50) & (pr_interval < 400)]
        # valid_qrs = qrs_width[(qrs_width > 20) & (qrs_width < 300)]
        
        # if len(valid_pr) > 0:
        #     global_pr_mean_val = np.mean(valid_pr)
        #     global_pr_std_val = np.std(valid_pr) if len(valid_pr) > 1 else 0.0
        # else:
        #     global_pr_mean_val = 160.0
        #     global_pr_std_val = 20.0
        
        # if len(valid_qrs) > 0:
        #     global_qrs_mean_val = np.mean(valid_qrs)
        #     global_qrs_std_val = np.std(valid_qrs) if len(valid_qrs) > 1 else 0.0
        # else:
        #     global_qrs_mean_val = 100.0
        #     global_qrs_std_val = 10.0
        
        # global_pr_mean[:] = global_pr_mean_val
        # global_pr_std[:] = global_pr_std_val
        # global_qrs_mean[:] = global_qrs_mean_val
        # global_qrs_std[:] = global_qrs_std_val
        
        # # Normalized features
        # pr_normalized = pr_interval / np.maximum(global_pr_mean, epsilon)
        # qrs_normalized = qrs_width / np.maximum(global_qrs_mean, epsilon)
        
    except Exception as e:
        # Use default values if delineation fails
        pr_interval[:] = 160.0
        qrs_width[:] = 100.0
        local_pr_mean[:] = 160.0
        local_qrs_mean[:] = 100.0
        global_pr_mean[:] = 160.0
        global_qrs_mean[:] = 100.0
        pr_normalized[:] = 1.0
        qrs_normalized[:] = 1.0
    
    # =========================================================================
    # Stack all features (n_beats, 38) — GROUPED BY SEMANTIC ROLE
    # =========================================================================

    all_features = np.stack([

        # ================================================================
        # G1. Local / Beat-to-Beat Rhythm Instability Features (SVEB trigger)
        # ================================================================
        pre_rr,               # [0]  Current RR interval
        post_rr,              # [1]  Next RR interval
        local_rr,             # [2]  Local mean RR (last 10 beats)
        local_std,            # [3]  Local RR standard deviation
        local_rmssd,          # [4]  Local RMSSD (short-term variability)
        rr_pre_div_cur,       # [5]  RR_{i-2} / RR_i
        pre_div_post,         # [7]  RR_i / RR_{i+1}
        # rr_post_div_cur,      # [6]  RR_{i+1} / RR_i
        # # ================================================================
        # # G2. Global Rhythm Context & Normalized Instability
        # # ================================================================
        # global_rr,            # [8]  Global RR mean
        # global_rr_std_arr,    # [9]  Global RR std
        # global_rmssd,         # [10] Global RMSSD
        # global_skew,          # [11] Global RR skewness
        # global_kurt,          # [12] Global RR kurtosis

        # pre_minus_global,     # [13] RR_i - global_RR
        # pre_div_global,       # [14] RR_i / global_RR
        # rr_cur_div_avg,       # [15] RR_i / global_RR (duplicate but explicit)

        # rr_global_normalized, # [16] Normalized RR
        # rr1_squared,          # [17] RR^2 (nonlinear sensitivity)
        # rr2_exp,              # [18] exp(RR)
        # rr3_exp_inv,          # [19] exp(1/RR)
        # rr4_log,              # [20] log(RR)

        # # ================================================================
        # # G3. AV Conduction / PR Interval Features (Supraventricular evidence)
        # # ================================================================
        # pr_interval,          # [21] PR interval (ms)
        # local_pr_mean,        # [22] Local PR mean
        # local_pr_std,         # [23] Local PR std
        # global_pr_mean,       # [24] Global PR mean
        # global_pr_std,        # [25] Global PR std
        # pr_normalized,        # [26] PR / global_PR_mean

        # # ================================================================
        # # G4. Ventricular Conduction / QRS Features
        # # ================================================================
        # qrs_width,            # [27] QRS width (ms)
        # local_qrs_mean,       # [28] Local QRS mean
        # local_qrs_std,        # [29] Local QRS std
        # global_qrs_mean,      # [30] Global QRS mean
        # global_qrs_std,       # [31] Global QRS std
        # qrs_normalized,       # [32] QRS / global_QRS_mean

        # # ================================================================
        # # G5. Temporal Look-ahead / Recovery Timing
        # # ================================================================
        # tr_half,              # [33] RR_{i+1} / 2
        # tr_quarter,           # [34] RR_{i+1} / 4
        # tr_half_norm,         # [35] Normalized half recovery
        # tr_quarter_norm,      # [36] Normalized quarter recovery
        # rr_post_div_avg,      # [37] RR_{i+1} / global_RR

    ], axis=1).astype(np.float32)

    
    return all_features

def compute_rr_features_opt2(
    ecg_signal: np.ndarray, 
    r_peaks: np.ndarray, 
    fs: int = 360
) -> np.ndarray:
    """
    Full RR + Morphological features (38 features)
    
    Uses NeuroKit2 for P/Q/S peak detection (slower but comprehensive)
    
    Groups:
        G1: RR instability (8)
        G2: Global rhythm (13)
        G3: PR interval (6)
        G4: QRS features (6)
        G5: Temporal recovery (5)
    
    Returns:
        features: (n_beats, 38) array
    """
    import neurokit2 as nk  # opt2에서만 사용
    
    n_beats = len(r_peaks)
    
    # Convert to ms units
    ms_factor = 1000.0 / fs
    
    # =========================================================================
    # Part 1: RR Interval Features (original 26 features)
    # =========================================================================
    
    # Initialize RR arrays
    pre_rr = np.zeros(n_beats, dtype=np.float32)
    post_rr = np.zeros(n_beats, dtype=np.float32)
    local_rr = np.zeros(n_beats, dtype=np.float32)
    local_rmssd = np.zeros(n_beats, dtype=np.float32)
    local_std = np.zeros(n_beats, dtype=np.float32)
    global_skew = np.zeros(n_beats, dtype=np.float32)
    global_kurt = np.zeros(n_beats, dtype=np.float32)
    
    rr_cur_div_avg = np.zeros(n_beats, dtype=np.float32)
    rr_post_div_cur = np.zeros(n_beats, dtype=np.float32)
    rr_pre_div_cur = np.zeros(n_beats, dtype=np.float32)
    rr_post_div_avg = np.zeros(n_beats, dtype=np.float32)
    
    tr_half = np.zeros(n_beats, dtype=np.float32)
    tr_quarter = np.zeros(n_beats, dtype=np.float32)
    tr_half_norm = np.zeros(n_beats, dtype=np.float32)
    tr_quarter_norm = np.zeros(n_beats, dtype=np.float32)
    
    rr_global_normalized = np.zeros(n_beats, dtype=np.float32)
    rr1_squared = np.zeros(n_beats, dtype=np.float32)
    rr2_exp = np.zeros(n_beats, dtype=np.float32)
    rr3_exp_inv = np.zeros(n_beats, dtype=np.float32)
    rr4_log = np.zeros(n_beats, dtype=np.float32)
    
    # Pre RR and Post RR
    for i in range(n_beats):
        if i > 0:
            pre_rr[i] = (r_peaks[i] - r_peaks[i-1]) * ms_factor
        else:
            if n_beats > 1:
                pre_rr[i] = (r_peaks[1] - r_peaks[0]) * ms_factor
            else:
                pre_rr[i] = 800.0
        
        if i < n_beats - 1:
            post_rr[i] = (r_peaks[i+1] - r_peaks[i]) * ms_factor
        else:
            if n_beats > 1:
                post_rr[i] = (r_peaks[-1] - r_peaks[-2]) * ms_factor
            else:
                post_rr[i] = 800.0
    
    # Local statistics
    for i in range(n_beats):
        start_idx = max(0, i - 9)
        window = pre_rr[start_idx:i+1]
        if len(window) > 0:
            local_rr[i] = np.mean(window)
            local_std[i] = np.std(window) if len(window) > 1 else 0.0
        else:
            local_rr[i] = pre_rr[i]
            local_std[i] = 0.0
    
    # Local RMSSD
    for i in range(n_beats):
        start = max(0, i - 9)
        rr_window = pre_rr[start:i+1]
        
        if len(rr_window) >= 2:
            diff_rr = np.diff(rr_window)
            local_rmssd[i] = np.sqrt(np.mean(diff_rr ** 2))
        else:
            local_rmssd[i] = 0.0
    
    # Global statistics
    valid_pre_rr = pre_rr[pre_rr > 50]
    
    if len(valid_pre_rr) > 1:
        global_rr_mean = np.mean(valid_pre_rr)
        global_rr_std = np.std(valid_pre_rr)
        global_rmssd_value = np.sqrt(np.mean(np.diff(valid_pre_rr) ** 2))
        
        if len(valid_pre_rr) >= 3:
            from scipy.stats import skew, kurtosis
            global_skew_value = float(skew(valid_pre_rr))
            global_kurt_value = float(kurtosis(valid_pre_rr)) if len(valid_pre_rr) >= 4 else 0.0
        else:
            global_skew_value = 0.0
            global_kurt_value = 0.0
    else:
        global_rr_mean = 800.0
        global_rr_std = 50.0
        global_rmssd_value = 50.0
        global_skew_value = 0.0
        global_kurt_value = 0.0
    
    global_rr = np.full(n_beats, global_rr_mean, dtype=np.float32)
    global_rr_std_arr = np.full(n_beats, global_rr_std, dtype=np.float32)
    global_rmssd = np.full(n_beats, global_rmssd_value, dtype=np.float32)
    global_skew[:] = global_skew_value
    global_kurt[:] = global_kurt_value
    
    # Derived features
    epsilon = 1.0
    
    pre_minus_global = pre_rr - global_rr
    pre_div_global = pre_rr / np.maximum(global_rr, epsilon)
    pre_div_post = pre_rr / np.maximum(post_rr, epsilon)
    
    # RR ratio features
    rr_cur_div_avg = pre_rr / np.maximum(global_rr, epsilon)
    rr_post_div_cur = post_rr / np.maximum(pre_rr, epsilon)
    
    for i in range(n_beats):
        if i > 0:
            prev_prev_rr = (r_peaks[i-1] - r_peaks[i-2]) * ms_factor if i > 1 else pre_rr[i]
            rr_pre_div_cur[i] = prev_prev_rr / max(pre_rr[i], epsilon)
        else:
            rr_pre_div_cur[i] = 1.0
    
    rr_post_div_avg = post_rr / np.maximum(global_rr, epsilon)
    
    # Temporal features
    for i in range(n_beats):
        if i < n_beats - 1:
            tr_half[i] = post_rr[i] / 2.0
            tr_quarter[i] = post_rr[i] / 4.0
            
            tr_half_norm[i] = tr_half[i] / max(global_rr_mean, epsilon)
            tr_quarter_norm[i] = tr_quarter[i] / max(global_rr_mean, epsilon)
        else:
            tr_half[i] = post_rr[i] / 2.0
            tr_quarter[i] = post_rr[i] / 4.0
            tr_half_norm[i] = tr_half[i] / max(global_rr_mean, epsilon)
            tr_quarter_norm[i] = tr_quarter[i] / max(global_rr_mean, epsilon)
    
    # RR_global transformations
    rr_global_normalized = pre_rr / np.maximum(global_rr, epsilon)
    rr_global_clipped = np.clip(rr_global_normalized, 0.01, 100.0)
    
    rr1_squared = rr_global_clipped ** 2
    rr2_exp = np.exp(np.clip(rr_global_clipped, -10, 10))
    rr3_exp_inv = np.exp(np.clip(1.0 / rr_global_clipped, -10, 10))
    rr4_log = np.log(rr_global_clipped)
    
    # =========================================================================
    # Part 2: Morphological Features (PR interval, QRS width) - NEW
    # =========================================================================
    
    # Initialize morphological arrays
    pr_interval = np.zeros(n_beats, dtype=np.float32)
    qrs_width = np.zeros(n_beats, dtype=np.float32)
    
    local_pr_mean = np.zeros(n_beats, dtype=np.float32)
    local_pr_std = np.zeros(n_beats, dtype=np.float32)
    local_qrs_mean = np.zeros(n_beats, dtype=np.float32)
    local_qrs_std = np.zeros(n_beats, dtype=np.float32)
    
    global_pr_mean = np.zeros(n_beats, dtype=np.float32)
    global_pr_std = np.zeros(n_beats, dtype=np.float32)
    global_qrs_mean = np.zeros(n_beats, dtype=np.float32)
    global_qrs_std = np.zeros(n_beats, dtype=np.float32)
    
    pr_normalized = np.zeros(n_beats, dtype=np.float32)
    qrs_normalized = np.zeros(n_beats, dtype=np.float32)
    
    try:
        # CRITICAL FIX: Ensure r_peaks is integer array and properly formatted
        r_peaks_int = np.asarray(r_peaks, dtype=np.int64).flatten()
        
        # Validate signal length
        if len(ecg_signal) < 100:
            raise ValueError("ECG signal too short for delineation")
        
        # Detect P peaks, Q peaks, S peaks using NeuroKit2
        _, waves_peak = nk.ecg_delineate(
            ecg_signal, 
            rpeaks=r_peaks_int,  # Use 'rpeaks' parameter name
            sampling_rate=fs,
            method="dwt"
        )
        
        # Extract peak arrays - handle both dict and DataFrame returns
        if isinstance(waves_peak, dict):
            p_peaks = waves_peak.get('ECG_P_Peaks', None)
            q_peaks = waves_peak.get('ECG_Q_Peaks', None)
            s_peaks = waves_peak.get('ECG_S_Peaks', None)
        else:
            # If it's a DataFrame
            p_peaks = waves_peak['ECG_P_Peaks'].values if 'ECG_P_Peaks' in waves_peak.columns else None
            q_peaks = waves_peak['ECG_Q_Peaks'].values if 'ECG_Q_Peaks' in waves_peak.columns else None
            s_peaks = waves_peak['ECG_S_Peaks'].values if 'ECG_S_Peaks' in waves_peak.columns else None
        
        # Filter out NaN values
        if p_peaks is not None:
            p_peaks = p_peaks[~np.isnan(p_peaks)].astype(np.int64)
        if q_peaks is not None:
            q_peaks = q_peaks[~np.isnan(q_peaks)].astype(np.int64)
        if s_peaks is not None:
            s_peaks = s_peaks[~np.isnan(s_peaks)].astype(np.int64)
        
        # ---------------------------------------------------------------------
        # Compute PR interval and QRS width for each beat
        # ---------------------------------------------------------------------
        for i in range(n_beats):
            r_idx = r_peaks_int[i]
            
            # PR interval: P peak to R peak (ms)
            if p_peaks is not None and len(p_peaks) > 0:
                valid_p = p_peaks[p_peaks < r_idx]
                if len(valid_p) > 0:
                    p_idx = valid_p[-1]
                    pr_interval[i] = (r_idx - p_idx) * ms_factor
                else:
                    pr_interval[i] = 160.0
            else:
                pr_interval[i] = 160.0
            
            # QRS width: Q peak to S peak (ms)
            if q_peaks is not None and s_peaks is not None and len(q_peaks) > 0 and len(s_peaks) > 0:
                valid_q = q_peaks[q_peaks <= r_idx]
                valid_s = s_peaks[s_peaks >= r_idx]
                
                if len(valid_q) > 0 and len(valid_s) > 0:
                    q_idx = valid_q[-1]
                    s_idx = valid_s[0]
                    qrs_width[i] = (s_idx - q_idx) * ms_factor
                else:
                    qrs_width[i] = 100.0
            else:
                qrs_width[i] = 100.0
        
        # ---------------------------------------------------------------------
        # Local statistics (last 10 beats window)
        # ---------------------------------------------------------------------
        for i in range(n_beats):
            start_idx = max(0, i - 9)
            
            # PR local stats
            pr_window = pr_interval[start_idx:i+1]
            if len(pr_window) > 0:
                local_pr_mean[i] = np.mean(pr_window)
                local_pr_std[i] = np.std(pr_window) if len(pr_window) > 1 else 0.0
            else:
                local_pr_mean[i] = pr_interval[i]
                local_pr_std[i] = 0.0
            
            # QRS local stats
            qrs_window = qrs_width[start_idx:i+1]
            if len(qrs_window) > 0:
                local_qrs_mean[i] = np.mean(qrs_window)
                local_qrs_std[i] = np.std(qrs_window) if len(qrs_window) > 1 else 0.0
            else:
                local_qrs_mean[i] = qrs_width[i]
                local_qrs_std[i] = 0.0
        
        # ---------------------------------------------------------------------
        # Global statistics
        # ---------------------------------------------------------------------
        valid_pr = pr_interval[(pr_interval > 50) & (pr_interval < 400)]
        valid_qrs = qrs_width[(qrs_width > 20) & (qrs_width < 300)]
        
        if len(valid_pr) > 0:
            global_pr_mean_val = np.mean(valid_pr)
            global_pr_std_val = np.std(valid_pr) if len(valid_pr) > 1 else 0.0
        else:
            global_pr_mean_val = 160.0
            global_pr_std_val = 20.0
        
        if len(valid_qrs) > 0:
            global_qrs_mean_val = np.mean(valid_qrs)
            global_qrs_std_val = np.std(valid_qrs) if len(valid_qrs) > 1 else 0.0
        else:
            global_qrs_mean_val = 100.0
            global_qrs_std_val = 10.0
        
        global_pr_mean[:] = global_pr_mean_val
        global_pr_std[:] = global_pr_std_val
        global_qrs_mean[:] = global_qrs_mean_val
        global_qrs_std[:] = global_qrs_std_val
        
        # Normalized features
        pr_normalized = pr_interval / np.maximum(global_pr_mean, epsilon)
        qrs_normalized = qrs_width / np.maximum(global_qrs_mean, epsilon)
        
    except Exception as e:
        # Use default values if delineation fails
        pr_interval[:] = 160.0
        qrs_width[:] = 100.0
        local_pr_mean[:] = 160.0
        local_qrs_mean[:] = 100.0
        global_pr_mean[:] = 160.0
        global_qrs_mean[:] = 100.0
        pr_normalized[:] = 1.0
        qrs_normalized[:] = 1.0
    
    # =========================================================================
    # Stack all features (n_beats, 38) — GROUPED BY SEMANTIC ROLE
    # =========================================================================

    all_features = np.stack([

        # ================================================================
        # G1. Local / Beat-to-Beat Rhythm Instability Features (SVEB trigger)
        # ================================================================
        pre_rr,               # [0]  Current RR interval
        post_rr,              # [1]  Next RR interval
        local_rr,             # [2]  Local mean RR (last 10 beats)
        local_std,            # [3]  Local RR standard deviation
        local_rmssd,          # [4]  Local RMSSD (short-term variability)
        rr_pre_div_cur,       # [5]  RR_{i-2} / RR_i
        pre_div_post,         # [7]  RR_i / RR_{i+1}
        rr_post_div_cur,      # [6]  RR_{i+1} / RR_i
        # ================================================================
        # G2. Global Rhythm Context & Normalized Instability
        # ================================================================
        global_rr,            # [8]  Global RR mean
        global_rr_std_arr,    # [9]  Global RR std
        global_rmssd,         # [10] Global RMSSD
        global_skew,          # [11] Global RR skewness
        global_kurt,          # [12] Global RR kurtosis

        pre_minus_global,     # [13] RR_i - global_RR
        pre_div_global,       # [14] RR_i / global_RR
        rr_cur_div_avg,       # [15] RR_i / global_RR (duplicate but explicit)

        rr_global_normalized, # [16] Normalized RR
        rr1_squared,          # [17] RR^2 (nonlinear sensitivity)
        rr2_exp,              # [18] exp(RR)
        rr3_exp_inv,          # [19] exp(1/RR)
        rr4_log,              # [20] log(RR)

        # ================================================================
        # G3. AV Conduction / PR Interval Features (Supraventricular evidence)
        # ================================================================
        pr_interval,          # [21] PR interval (ms)
        local_pr_mean,        # [22] Local PR mean
        local_pr_std,         # [23] Local PR std
        global_pr_mean,       # [24] Global PR mean
        global_pr_std,        # [25] Global PR std
        pr_normalized,        # [26] PR / global_PR_mean

        # ================================================================
        # G4. Ventricular Conduction / QRS Features
        # ================================================================
        qrs_width,            # [27] QRS width (ms)
        local_qrs_mean,       # [28] Local QRS mean
        local_qrs_std,        # [29] Local QRS std
        global_qrs_mean,      # [30] Global QRS mean
        global_qrs_std,       # [31] Global QRS std
        qrs_normalized,       # [32] QRS / global_QRS_mean

        # ================================================================
        # G5. Temporal Look-ahead / Recovery Timing
        # ================================================================
        tr_half,              # [33] RR_{i+1} / 2
        tr_quarter,           # [34] RR_{i+1} / 4
        tr_half_norm,         # [35] Normalized half recovery
        tr_quarter_norm,      # [36] Normalized quarter recovery
        rr_post_div_avg,      # [37] RR_{i+1} / global_RR

    ], axis=1).astype(np.float32)

    
    return all_features


def compute_rr_features_opt3(
    ecg_signal: np.ndarray, 
    r_peaks: np.ndarray, 
    fs: int = 360
) -> np.ndarray:
    """
    Basic RR features with global context (7 features)
    
    Features:
        [0] pre_rr         - Current RR interval (ms)
        [1] post_rr        - Next RR interval (ms)
        [2] local_rr       - Local mean RR (last 10 beats)
        [3] pre_div_post   - RR_i / RR_{i+1}
        [4] global_rr      - Global RR mean
        [5] pre_minus_global- RR_i - global_RR
        [6] pre_div_global - RR_i / global_RR
    
    Returns:
        features: (n_beats, 7) array
    """
    n_beats = len(r_peaks)
    
    # Convert to ms units
    ms_factor = 1000.0 / fs
    
    # =========================================================================
    # Part 1: RR Interval Features (original 26 features)
    # =========================================================================
    
    # Initialize RR arrays
    pre_rr = np.zeros(n_beats, dtype=np.float32)
    post_rr = np.zeros(n_beats, dtype=np.float32)
    local_rr = np.zeros(n_beats, dtype=np.float32)
    local_rmssd = np.zeros(n_beats, dtype=np.float32)
    local_std = np.zeros(n_beats, dtype=np.float32)
    global_skew = np.zeros(n_beats, dtype=np.float32)
    global_kurt = np.zeros(n_beats, dtype=np.float32)
    
    rr_cur_div_avg = np.zeros(n_beats, dtype=np.float32)
    rr_post_div_cur = np.zeros(n_beats, dtype=np.float32)
    rr_pre_div_cur = np.zeros(n_beats, dtype=np.float32)
    rr_post_div_avg = np.zeros(n_beats, dtype=np.float32)
    
    tr_half = np.zeros(n_beats, dtype=np.float32)
    tr_quarter = np.zeros(n_beats, dtype=np.float32)
    tr_half_norm = np.zeros(n_beats, dtype=np.float32)
    tr_quarter_norm = np.zeros(n_beats, dtype=np.float32)
    
    rr_global_normalized = np.zeros(n_beats, dtype=np.float32)
    rr1_squared = np.zeros(n_beats, dtype=np.float32)
    rr2_exp = np.zeros(n_beats, dtype=np.float32)
    rr3_exp_inv = np.zeros(n_beats, dtype=np.float32)
    rr4_log = np.zeros(n_beats, dtype=np.float32)
    
    # Pre RR and Post RR
    for i in range(n_beats):
        if i > 0:
            pre_rr[i] = (r_peaks[i] - r_peaks[i-1]) * ms_factor
        else:
            if n_beats > 1:
                pre_rr[i] = (r_peaks[1] - r_peaks[0]) * ms_factor
            else:
                pre_rr[i] = 800.0
        
        if i < n_beats - 1:
            post_rr[i] = (r_peaks[i+1] - r_peaks[i]) * ms_factor
        else:
            if n_beats > 1:
                post_rr[i] = (r_peaks[-1] - r_peaks[-2]) * ms_factor
            else:
                post_rr[i] = 800.0
    
    # Local statistics
    for i in range(n_beats):
        start_idx = max(0, i - 9)
        window = pre_rr[start_idx:i+1]
        if len(window) > 0:
            local_rr[i] = np.mean(window)
            local_std[i] = np.std(window) if len(window) > 1 else 0.0
        else:
            local_rr[i] = pre_rr[i]
            local_std[i] = 0.0
    
    # Local RMSSD
    for i in range(n_beats):
        start = max(0, i - 9)
        rr_window = pre_rr[start:i+1]
        
        if len(rr_window) >= 2:
            diff_rr = np.diff(rr_window)
            local_rmssd[i] = np.sqrt(np.mean(diff_rr ** 2))
        else:
            local_rmssd[i] = 0.0
    
    # Global statistics
    valid_pre_rr = pre_rr[pre_rr > 50]
    
    if len(valid_pre_rr) > 1:
        global_rr_mean = np.mean(valid_pre_rr)
        global_rr_std = np.std(valid_pre_rr)
        global_rmssd_value = np.sqrt(np.mean(np.diff(valid_pre_rr) ** 2))
        
        if len(valid_pre_rr) >= 3:
            from scipy.stats import skew, kurtosis
            global_skew_value = float(skew(valid_pre_rr))
            global_kurt_value = float(kurtosis(valid_pre_rr)) if len(valid_pre_rr) >= 4 else 0.0
        else:
            global_skew_value = 0.0
            global_kurt_value = 0.0
    else:
        global_rr_mean = 800.0
        global_rr_std = 50.0
        global_rmssd_value = 50.0
        global_skew_value = 0.0
        global_kurt_value = 0.0
    
    global_rr = np.full(n_beats, global_rr_mean, dtype=np.float32)
    global_rr_std_arr = np.full(n_beats, global_rr_std, dtype=np.float32)
    global_rmssd = np.full(n_beats, global_rmssd_value, dtype=np.float32)
    global_skew[:] = global_skew_value
    global_kurt[:] = global_kurt_value
    
    # Derived features
    epsilon = 1.0
    
    pre_minus_global = pre_rr - global_rr
    pre_div_global = pre_rr / np.maximum(global_rr, epsilon)
    pre_div_post = pre_rr / np.maximum(post_rr, epsilon)
    
    # RR ratio features
    rr_cur_div_avg = pre_rr / np.maximum(global_rr, epsilon)
    rr_post_div_cur = post_rr / np.maximum(pre_rr, epsilon)
    
    for i in range(n_beats):
        if i > 0:
            prev_prev_rr = (r_peaks[i-1] - r_peaks[i-2]) * ms_factor if i > 1 else pre_rr[i]
            rr_pre_div_cur[i] = prev_prev_rr / max(pre_rr[i], epsilon)
        else:
            rr_pre_div_cur[i] = 1.0
    
    rr_post_div_avg = post_rr / np.maximum(global_rr, epsilon)
    
    # Temporal features
    for i in range(n_beats):
        if i < n_beats - 1:
            tr_half[i] = post_rr[i] / 2.0
            tr_quarter[i] = post_rr[i] / 4.0
            
            tr_half_norm[i] = tr_half[i] / max(global_rr_mean, epsilon)
            tr_quarter_norm[i] = tr_quarter[i] / max(global_rr_mean, epsilon)
        else:
            tr_half[i] = post_rr[i] / 2.0
            tr_quarter[i] = post_rr[i] / 4.0
            tr_half_norm[i] = tr_half[i] / max(global_rr_mean, epsilon)
            tr_quarter_norm[i] = tr_quarter[i] / max(global_rr_mean, epsilon)
    
    # RR_global transformations
    rr_global_normalized = pre_rr / np.maximum(global_rr, epsilon)
    rr_global_clipped = np.clip(rr_global_normalized, 0.01, 100.0)
    
    rr1_squared = rr_global_clipped ** 2
    rr2_exp = np.exp(np.clip(rr_global_clipped, -10, 10))
    rr3_exp_inv = np.exp(np.clip(1.0 / rr_global_clipped, -10, 10))
    rr4_log = np.log(rr_global_clipped)
    
    # =========================================================================
    # Part 2: Morphological Features (PR interval, QRS width) - NEW
    # =========================================================================
    
    # Initialize morphological arrays
    pr_interval = np.zeros(n_beats, dtype=np.float32)
    qrs_width = np.zeros(n_beats, dtype=np.float32)
    
    local_pr_mean = np.zeros(n_beats, dtype=np.float32)
    local_pr_std = np.zeros(n_beats, dtype=np.float32)
    local_qrs_mean = np.zeros(n_beats, dtype=np.float32)
    local_qrs_std = np.zeros(n_beats, dtype=np.float32)
    
    global_pr_mean = np.zeros(n_beats, dtype=np.float32)
    global_pr_std = np.zeros(n_beats, dtype=np.float32)
    global_qrs_mean = np.zeros(n_beats, dtype=np.float32)
    global_qrs_std = np.zeros(n_beats, dtype=np.float32)
    
    pr_normalized = np.zeros(n_beats, dtype=np.float32)
    qrs_normalized = np.zeros(n_beats, dtype=np.float32)
    
    try:
    #     #
        # ---------------------------------------------------------------------
        # Local statistics (last 10 beats window)
        # ---------------------------------------------------------------------
        for i in range(n_beats):
            start_idx = max(0, i - 9)
            
            # PR local stats
            pr_window = pr_interval[start_idx:i+1]
            if len(pr_window) > 0:
                local_pr_mean[i] = np.mean(pr_window)
                local_pr_std[i] = np.std(pr_window) if len(pr_window) > 1 else 0.0
            else:
                local_pr_mean[i] = pr_interval[i]
                local_pr_std[i] = 0.0
            
            # QRS local stats
            qrs_window = qrs_width[start_idx:i+1]
            if len(qrs_window) > 0:
                local_qrs_mean[i] = np.mean(qrs_window)
                local_qrs_std[i] = np.std(qrs_window) if len(qrs_window) > 1 else 0.0
            else:
                local_qrs_mean[i] = qrs_width[i]
                local_qrs_std[i] = 0.0
        
        # ---------------------------------------------------------------------

    except Exception as e:
        # Use default values if delineation fails
        pr_interval[:] = 160.0
        qrs_width[:] = 100.0
        local_pr_mean[:] = 160.0
        local_qrs_mean[:] = 100.0
        global_pr_mean[:] = 160.0
        global_qrs_mean[:] = 100.0
        pr_normalized[:] = 1.0
        qrs_normalized[:] = 1.0
    
    # =========================================================================
    # Stack all features (n_beats, 38) — GROUPED BY SEMANTIC ROLE
    # =========================================================================

    all_features = np.stack([


        pre_rr,               # [0]  Current RR interval
        post_rr,              # [1]  Next RR interval
        local_rr,             # [2]  Local mean RR (last 10 beats)
        pre_div_post,         # [7]  RR_i / RR_{i+1}
        global_rr,            # [8]  Global RR mean
        pre_minus_global,     # [13] RR_i - global_RR
        pre_div_global,       # [14] RR_i / global_RR
        # rr_cur_div_avg,       # [15] RR_i / global_RR (duplicate but explicit)

   
    ], axis=1).astype(np.float32)

    
    return all_features


def compute_rr_features_opt4(
    ecg_signal: np.ndarray, 
    r_peaks: np.ndarray, 
    fs: int = 360
) -> np.ndarray:
    """
    Numerically Stable RR features (7 features)
    
    수치적 안정성을 최대화한 feature 설계:
    - 뺄셈 연산 제거 (cancellation error 방지)
    - 비율(ratio) 기반으로 스케일 불변
    - bounded range로 정규화
    - exp/log 변환 제거
    
    Features:
        [0] pre_rr_norm     - 정규화된 현재 RR (global mean으로 나눔, clipped)
        [1] post_rr_norm    - 정규화된 다음 RR
        [2] local_rr_norm   - 정규화된 local mean RR
        [3] pre_div_post    - RR_i / RR_{i+1} (연속 비율)
        [4] pre_div_local   - RR_i / local_mean (local 대비 비율)
        [5] local_cv        - Local CV (std/mean, 스케일 불변 변동성)
        [6] rr_stability    - 연속 3개 RR의 안정성 지표
    
    Returns:
        features: (n_beats, 7) array
    """
    n_beats = len(r_peaks)
    
    # Convert to ms units
    ms_factor = 1000.0 / fs
    
    # Initialize arrays
    pre_rr = np.zeros(n_beats, dtype=np.float32)
    post_rr = np.zeros(n_beats, dtype=np.float32)
    local_rr = np.zeros(n_beats, dtype=np.float32)
    local_std = np.zeros(n_beats, dtype=np.float32)
    
    # =========================================================================
    # Step 1: 기본 RR 간격 계산
    # =========================================================================
    for i in range(n_beats):
        # Pre RR (현재 RR)
        if i > 0:
            pre_rr[i] = (r_peaks[i] - r_peaks[i-1]) * ms_factor
        else:
            pre_rr[i] = 800.0  # default
        
        # Post RR (다음 RR)
        if i < n_beats - 1:
            post_rr[i] = (r_peaks[i+1] - r_peaks[i]) * ms_factor
        else:
            post_rr[i] = pre_rr[i]  # 마지막은 현재와 동일
        
        # Local RR mean & std (지난 10개 beat)
        start = max(0, i - 9)
        rr_window = pre_rr[start:i+1]
        
        if len(rr_window) >= 1:
            local_rr[i] = np.mean(rr_window)
            local_std[i] = np.std(rr_window) if len(rr_window) >= 2 else 0.0
        else:
            local_rr[i] = 800.0
            local_std[i] = 0.0
    
    # =========================================================================
    # Step 2: Global 통계량 계산 (수치적으로 안정된 방식)
    # =========================================================================
    # 극단값 제외한 robust mean
    valid_rr = pre_rr[(pre_rr > 200) & (pre_rr < 2000)]  # 200ms~2000ms 범위
    
    if len(valid_rr) > 0:
        global_rr_mean = np.median(valid_rr)  # mean 대신 median 사용 (robust)
    else:
        global_rr_mean = 800.0
    
    # Epsilon: global mean의 1% (스케일에 맞게 조정)
    epsilon = max(global_rr_mean * 0.01, 1.0)
    
    # =========================================================================
    # Step 3: 수치적으로 안정된 Feature 계산
    # =========================================================================
    
    # [0] pre_rr_norm: 정규화된 현재 RR (0.1 ~ 10 범위로 clip)
    pre_rr_norm = np.clip(pre_rr / global_rr_mean, 0.1, 10.0)
    
    # [1] post_rr_norm: 정규화된 다음 RR
    post_rr_norm = np.clip(post_rr / global_rr_mean, 0.1, 10.0)
    
    # [2] local_rr_norm: 정규화된 local mean
    local_rr_norm = np.clip(local_rr / global_rr_mean, 0.1, 10.0)
    
    # [3] pre_div_post: 연속 RR 비율 (0.1 ~ 10 범위로 clip)
    pre_div_post = np.clip(pre_rr / np.maximum(post_rr, epsilon), 0.1, 10.0)
    
    # [4] pre_div_local: 현재 RR / local mean (local 대비 비율)
    pre_div_local = np.clip(pre_rr / np.maximum(local_rr, epsilon), 0.1, 10.0)
    
    # [5] local_cv: Coefficient of Variation (std/mean, 0~1 범위)
    # CV는 스케일 불변이고 수치적으로 안정
    local_cv = np.clip(local_std / np.maximum(local_rr, epsilon), 0.0, 1.0)
    
    # [6] rr_stability: 연속 3개 RR의 안정성 (max/min ratio, 1~10 범위)
    # 값이 1에 가까울수록 안정, 클수록 불안정
    rr_stability = np.ones(n_beats, dtype=np.float32)
    for i in range(1, n_beats - 1):
        rr_triplet = np.array([pre_rr[i-1], pre_rr[i], post_rr[i]])
        rr_triplet = rr_triplet[rr_triplet > epsilon]
        if len(rr_triplet) >= 2:
            rr_stability[i] = np.clip(np.max(rr_triplet) / np.min(rr_triplet), 1.0, 10.0)
    
    # =========================================================================
    # Stack features
    # =========================================================================
    all_features = np.stack([
        pre_rr_norm,      # [0] 정규화된 현재 RR
        post_rr_norm,     # [1] 정규화된 다음 RR
        local_rr_norm,    # [2] 정규화된 local mean
        pre_div_post,     # [3] 연속 RR 비율
        pre_div_local,    # [4] local 대비 비율
        local_cv,         # [5] Local CV (변동성)
        rr_stability,     # [6] 연속 RR 안정성
    ], axis=1).astype(np.float32)
    
    return all_features


# =============================================================================
# Beat Extraction
# =============================================================================

def extract_beats_and_rr_from_records_single(record_list: list, base_path: str,
                                       valid_leads: list, out_len: int,
                                       split_name: str) -> tuple:
    all_data = []
    all_labels_id = []
    all_rr_features = []
    all_patient_ids = []
    all_indexes = []
    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}

    skipped = 0
    for rec in tqdm(record_list, desc=f"Extracting {split_name}"):
        try:
            ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
            sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
        except Exception as e:
            print(f"Warning: {rec} read failed - {e}")
            continue

        patient_id = patient_to_id[rec]
        fs = int(meta['fs'])
        sig_names = meta['sig_name']

        ch_idx = None
        for lead in valid_leads:
            if lead in sig_names:
                ch_idx = sig_names.index(lead)
                break

        if ch_idx is None:
            continue

        x = sig[:, ch_idx].astype(np.float32)
        x_filtered = remove_baseline_bandpass(x, fs=fs)

        r_peaks = ann.sample
        rr_func = get_rr_feature_function(RR_FEATURE_OPTION)
        rr_features = rr_func(x_filtered, r_peaks, fs)

        pre = int(round(360 * fs / 360.0))
        post = int(round(360 * fs / 360.0))

        for idx, (center, symbol) in enumerate(zip(ann.sample, ann.symbol)):
            grp = LABEL_GROUP_MAP.get(symbol, None)

            if grp is None or grp not in CLASSES:
                continue

            start = center - pre
            end = center + post

            if start < 0 or end > len(x_filtered):
                skipped += 1
                continue

            seg = x_filtered[start:end]
            seg = (seg - seg.mean()) / (seg.std() + 1e-8)
            seg_resampled = resample_to_len(seg, out_len)

            all_data.append(seg_resampled)
            all_labels_id.append(LABEL_TO_ID[grp])
            all_rr_features.append(rr_features[idx])
            all_patient_ids.append(patient_id)
            all_indexes.append(idx)

    print(f"{split_name} - Skipped: {skipped}, Extracted: {len(all_data)}")

    return (np.array(all_data, dtype=np.float32),
            np.array(all_labels_id, dtype=np.int64),
            np.array(all_rr_features, dtype=np.float32),
            np.array(all_patient_ids, dtype=np.int64),
            np.array(all_indexes, dtype=np.int64))

from typing import Tuple

def process_single_record(args) -> Tuple:
    rec, base_path, valid_leads, out_len, patient_id = args

    data = []
    labels = []
    rr_feats = []
    pids = []
    indexes = []
    skipped = 0

    try:
        ann = wfdb.rdann(os.path.join(base_path, rec), 'atr')
        sig, meta = wfdb.rdsamp(os.path.join(base_path, rec))
    except Exception as e:
        print(f"Warning: {rec} read failed - {e}")
        return data, labels, rr_feats, pids, indexes, skipped

    fs = int(meta['fs'])
    sig_names = meta['sig_name']

    ch_idx = None
    for lead in valid_leads:
        if lead in sig_names:
            ch_idx = sig_names.index(lead)
            break
    if ch_idx is None:
        return data, labels, rr_feats, pids, indexes, skipped

    x = sig[:, ch_idx].astype(np.float32)
    x_filtered = remove_baseline_bandpass(x, fs=fs)

    r_peaks = ann.sample
    rr_func = get_rr_feature_function(RR_FEATURE_OPTION)
    rr_features = rr_func(x_filtered, r_peaks, fs)

    pre = int(round(360 * fs / 360.0))
    post = int(round(360 * fs / 360.0))

    for idx, (center, symbol) in enumerate(zip(ann.sample, ann.symbol)):
        grp = LABEL_GROUP_MAP.get(symbol, None)
        if grp is None or grp not in CLASSES:
            continue

        start = center - pre
        end = center + post
        if start < 0 or end > len(x_filtered):
            skipped += 1
            continue

        seg = x_filtered[start:end]
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        seg_resampled = resample_to_len(seg, out_len)

        data.append(seg_resampled)
        labels.append(LABEL_TO_ID[grp])
        rr_feats.append(rr_features[idx])
        pids.append(patient_id)
        indexes.append(idx)

    return data, labels, rr_feats, pids, indexes, skipped


def extract_beats_and_rr_from_records(
    record_list: list,
    base_path: str,
    valid_leads: list,
    out_len: int,
    split_name: str,
    n_jobs: int = None
) -> tuple:

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)


    patient_to_id = {rec: idx for idx, rec in enumerate(sorted(record_list))}

    args = [
        (rec, base_path, valid_leads, out_len, patient_to_id[rec])
        for rec in record_list
    ]

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(delayed(process_single_record)(a) for a in args)

    all_data, all_labels, all_rr, all_pids, all_idx = [], [], [], [], []
    skipped_total = 0

    for data, labels, rr, pids, idxs, skipped in results:
        all_data.extend(data)
        all_labels.extend(labels)
        all_rr.extend(rr)
        all_pids.extend(pids)
        all_idx.extend(idxs)
        skipped_total += skipped

    print(f"{split_name} - Skipped: {skipped_total}, Extracted: {len(all_data)}")

    return (
        np.asarray(all_data, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_rr, dtype=np.float32),
        np.asarray(all_pids, dtype=np.int64),
        np.asarray(all_idx, dtype=np.int64),
    )


# =============================================================================
# Loss function
# =============================================================================


class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0, beta=0.999, class_counts=None,
                 device='cuda', reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.device = device

        if class_counts is not None:
            if isinstance(class_counts, list):
                class_counts = torch.tensor(class_counts, dtype=torch.float32)

            # effective number of samples
            effective_num = 1.0 - torch.pow(self.beta, class_counts)
            weights = (1.0 - self.beta) / effective_num

            # optional normalization (논문에서 권장)
            weights = weights / weights.sum() * len(class_counts)

            self.weights = weights.to(device)
            print(f"CB-Focal weights: {self.weights.cpu().numpy()}")
        else:
            self.weights = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_term * log_pt + torch.pow(1.0 - pt,  self.gamma+1)

        if self.weights is not None:
            loss = loss * self.weights[targets]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =============================================================================
# Dataset Caching System
# =============================================================================

def _compute_cache_hash(record_list: list, out_len: int, valid_leads: list) -> str:
    """
    record 리스트, out_len, leads, RR option을 기반으로 고유 해시 생성
    """
    cache_key = {
        'records': sorted(record_list),
        'out_len': out_len,
        'valid_leads': valid_leads,
        'rr_option': RR_FEATURE_OPTION,  # RR 옵션도 캐시 키에 포함
    }
    key_str = json.dumps(cache_key, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _get_cache_paths(split_name: str, cache_hash: str, cache_dir: str = CACHE_DIR) -> dict:
    """
    캐시 파일 경로들 반환
    """
    os.makedirs(cache_dir, exist_ok=True)
    prefix = f"{split_name}_{cache_hash}"
    return {
        'data': os.path.join(cache_dir, f"{prefix}_data.npy"),
        'labels': os.path.join(cache_dir, f"{prefix}_labels.npy"),
        'rr': os.path.join(cache_dir, f"{prefix}_rr.npy"),
        'patient_ids': os.path.join(cache_dir, f"{prefix}_patient_ids.npy"),
        'sample_ids': os.path.join(cache_dir, f"{prefix}_sample_ids.npy"),
        'meta': os.path.join(cache_dir, f"{prefix}_meta.json"),
    }


def _is_cache_valid(cache_paths: dict, record_list: list, out_len: int) -> bool:
    """
    캐시 유효성 검사:
    - 모든 파일 존재 확인
    - 메타데이터 비교 (record 리스트, shape 등)
    """
    # 파일 존재 확인
    for key, path in cache_paths.items():
        if not os.path.exists(path):
            return False
    
    try:
        # 메타데이터 로드 및 비교
        with open(cache_paths['meta'], 'r') as f:
            meta = json.load(f)
        
        # record 리스트 일치 확인
        if sorted(meta.get('records', [])) != sorted(record_list):
            print("  Cache invalid: record list mismatch")
            return False
        
        # out_len 확인
        if meta.get('out_len') != out_len:
            print("  Cache invalid: out_len mismatch")
            return False
        
        # 샘플 shape 검증 (첫 번째 샘플)
        data_sample = np.load(cache_paths['data'], mmap_mode='r')
        if len(data_sample) == 0:
            print("  Cache invalid: empty data")
            return False
        
        if data_sample.shape[1] != out_len:
            print(f"  Cache invalid: signal length mismatch ({data_sample.shape[1]} vs {out_len})")
            return False
        
        # RR feature dimension 확인
        rr_sample = np.load(cache_paths['rr'], mmap_mode='r')
        if len(rr_sample) > 0 and rr_sample.shape[1] != meta.get('rr_dim', 7):
            print(f"  Cache invalid: RR dimension mismatch")
            return False
        
        return True
        
    except Exception as e:
        print(f"  Cache validation error: {e}")
        return False


def _save_cache(cache_paths: dict, data: tuple, record_list: list, out_len: int, rr_dim: int):
    """
    데이터셋을 numpy 파일로 저장
    """
    data_arr, labels_arr, rr_arr, patient_ids, sample_ids = data
    
    np.save(cache_paths['data'], data_arr)
    np.save(cache_paths['labels'], labels_arr)
    np.save(cache_paths['rr'], rr_arr)
    np.save(cache_paths['patient_ids'], patient_ids)
    np.save(cache_paths['sample_ids'], sample_ids)
    
    # 메타데이터 저장
    meta = {
        'records': sorted(record_list),
        'out_len': out_len,
        'rr_dim': rr_dim,
        'n_samples': len(data_arr),
        'data_shape': list(data_arr.shape),
        'rr_shape': list(rr_arr.shape),
        'label_distribution': {int(k): int(v) for k, v in zip(*np.unique(labels_arr, return_counts=True))},
    }
    
    with open(cache_paths['meta'], 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Cache saved: {len(data_arr)} samples")


def _load_cache(cache_paths: dict) -> tuple:
    """
    캐시에서 데이터 로드
    """
    data = np.load(cache_paths['data'])
    labels = np.load(cache_paths['labels'])
    rr = np.load(cache_paths['rr'])
    patient_ids = np.load(cache_paths['patient_ids'])
    sample_ids = np.load(cache_paths['sample_ids'])
    
    return data, labels, rr, patient_ids, sample_ids


def load_or_extract_data(
    record_list: list,
    base_path: str,
    valid_leads: list,
    out_len: int,
    split_name: str,
    n_jobs: int = None,
    cache_dir: str = CACHE_DIR,
    force_reprocess: bool = False
) -> tuple:
    """
    캐시 시스템을 사용한 데이터 로드/추출
    
    1. 캐시 해시 계산
    2. 캐시 유효성 검사
    3. 유효하면 로드, 아니면 추출 후 저장
    
    Args:
        record_list: 레코드 ID 리스트
        base_path: 데이터 경로
        valid_leads: 유효한 리드 리스트
        out_len: 출력 시그널 길이
        split_name: 데이터셋 이름 (Train/Valid/Test)
        n_jobs: 병렬 처리 작업 수
        cache_dir: 캐시 저장 디렉토리
        force_reprocess: True면 캐시 무시하고 재처리
    
    Returns:
        (data, labels, rr_features, patient_ids, sample_ids)
    """
    cache_hash = _compute_cache_hash(record_list, out_len, valid_leads)
    cache_paths = _get_cache_paths(split_name, cache_hash, cache_dir)
    
    print(f"\n[{split_name}] Cache check (hash: {cache_hash})")
    
    # 캐시 유효성 검사
    if not force_reprocess and _is_cache_valid(cache_paths, record_list, out_len):
        print(f"  Loading from cache...")
        data, labels, rr, patient_ids, sample_ids = _load_cache(cache_paths)
        print(f"  Loaded {len(data)} samples from cache")
        return data, labels, rr, patient_ids, sample_ids
    
    # 캐시 없거나 유효하지 않음 -> 전처리 실행
    print(f"  Cache not found or invalid. Processing...")
    data, labels, rr, patient_ids, sample_ids = extract_beats_and_rr_from_records(
        record_list=record_list,
        base_path=base_path,
        valid_leads=valid_leads,
        out_len=out_len,
        split_name=split_name,
        n_jobs=n_jobs
    )
    
    # 캐시 저장
    if len(data) > 0:
        rr_dim = rr.shape[1] if len(rr.shape) > 1 else 0
        _save_cache(cache_paths, (data, labels, rr, patient_ids, sample_ids), 
                   record_list, out_len, rr_dim)
    
    return data, labels, rr, patient_ids, sample_ids
