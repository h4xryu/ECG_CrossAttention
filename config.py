
import os
from datetime import datetime

# =============================================================================
# 실험 설정 
# =============================================================================

# 실험 이름 형식: "{모델타입}" 또는 "{모델타입}-{추가설명}"
# 예시:
#   "baseline"              - ECG only 기본
#   "baseline-dropout05"    - ECG only + dropout 0.5 실험
#   "naive_concatenate"     - ECG + RR 단순 결합
#   "cross_attention-v2"    - Cross-Attention 버전2
#
# 지원 모델: baseline, naive_concatenate, cross_attention
EXP_NAME = "baseline_A1_at"


def get_model_type(exp_name: str) -> str:
    """
    EXP_NAME에서 모델 타입 추출
    "baseline-xxx" -> "baseline"
    "cross_attention-v2" -> "cross_attention"
    """
    model_types = ["baseline", "naive_concatenate", "cross_attention"]
    
    for mt in model_types:
        if mt in exp_name:
            return mt
    
    raise ValueError(f"Unknown model type in EXP_NAME: '{exp_name}'. "
                     f"Must contain one of: {model_types}")

# =============================================================================
# 분석용 설정 (main_analysis.py에서 사용)
# =============================================================================
# 분석할 실험 폴더 (예: experiment_20260115_143000_baseline)
ANALYSIS_EXP_DIR = "./ECG_Results/experiment_20260115_143000_baseline"
# 분석할 모델 파일명 (best_model_auprc, best_model_wf1, best_model_wrecall, 또는 epoch_30)
ANALYSIS_MODEL_TYPE = "best_model_auprc"

# =============================================================================
# Paths
# =============================================================================
DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH = './ECG_Results/'

# =============================================================================
# Hyperparameters
# =============================================================================
BATCH_SIZE = 1024
EPOCHS = 50
LR = 0.0001
WEIGHT_DECAY = 1e-3
SEED = 1234
POLY1_EPS = 0.0
POLY2_EPS = 0.0

# =============================================================================
# RR Feature 옵션
# =============================================================================
# opt1: 7개 (basic) - pre_rr, post_rr, local_rr, local_std, local_rmssd, rr_pre_div_cur, pre_div_post
# opt2: 38개 (full)  - 전체 RR + morphology (PR, QRS) with NeuroKit2 (느림)
# opt3: 7개 (basic2) - pre_rr, post_rr, local_rr, pre_div_post, global_rr, pre_minus_global, pre_div_global
RR_FEATURE_OPTION = "opt3"

# 옵션별 feature 수 (자동 설정용)
RR_FEATURE_DIMS = {
    "opt1": 7,
    "opt2": 38,
    "opt3": 7,
    "opt4": 7,  # numerically stable features
}

# =============================================================================
# Model Architecture (공통)
# =============================================================================
MODEL_CONFIG = {
    'in_channels': 1,
    'out_ch': 180,
    'mid_ch': 30,
    'num_heads': 9,
    'n_rr': RR_FEATURE_DIMS[RR_FEATURE_OPTION],  # 자동 설정
}

# =============================================================================
# ECG Parameters
# =============================================================================
FS_TARGET = 360
OUT_LEN = 720
VALID_LEADS = ['MLII', 'V1', 'V2', 'V4', 'V5']

# =============================================================================
# Classes (AAMI 4-class)
# =============================================================================
CLASSES = ['N', 'S', 'V', 'F']
LABEL_TO_ID = {'N': 0, 'S': 1, 'V': 2, 'F': 3}
ID_TO_LABEL = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}

LABEL_GROUP_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
}

# =============================================================================
# Data Split (Chazal)
# =============================================================================
DS1_TRAIN = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119',
    '122', '201', '203', '209', '215', '223', '230', '208' ,           
]
DS1_VALID = [
    '114', '124', '205', '207', '220'
]




DS2_TEST = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]


def create_experiment_dir(output_path=OUTPUT_PATH, exp_name=EXP_NAME):
    """실험 결과 저장 폴더 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_path, f'experiment_{timestamp}_{exp_name}')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'correct'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'incorrect'), exist_ok=True)
    return exp_dir
