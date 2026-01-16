# main_analysis.py - 학습된 모델 성능 평가 스크립트
# 사용법: config.py에서 ANALYSIS_EXP_DIR, ANALYSIS_MODEL_TYPE 설정 후 실행

import os
from collections import Counter
import torch
from torch.utils.data import DataLoader

from config import (
    DATA_PATH, BATCH_SIZE, SEED, VALID_LEADS, OUT_LEN, CLASSES,
    DS1_TRAIN, DS1_VALID, DS2_TEST, MODEL_CONFIG,
    ANALYSIS_EXP_DIR, ANALYSIS_MODEL_TYPE
)
from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset
from test import (
    evaluate, calculate_metrics, print_metrics,
    save_results_csv, save_confusion_matrix,
    plot_ambiguous_sveb, plot_ambiguous_correct_sveb, plot_ambiguous_n_as_s,
    analyze_low_margin_sveb, margin_statistics
)

# =============================================================================
# 초기화
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(SEED)

# =============================================================================
# 경로 설정
# =============================================================================
exp_dir = ANALYSIS_EXP_DIR

# 실험 이름 추출 (experiment_20260115_143000_baseline -> baseline)
exp_name_parts = os.path.basename(exp_dir).split("_")
if len(exp_name_parts) >= 4:
    exp_name = "_".join(exp_name_parts[3:])
else:
    exp_name = "baseline"

# 모델 파일 경로 결정
if ANALYSIS_MODEL_TYPE.startswith("best_model"):
    weights_path = os.path.join(exp_dir, f"{ANALYSIS_MODEL_TYPE}_{exp_name}.pth")
elif ANALYSIS_MODEL_TYPE.startswith("epoch"):
    epoch_num = ANALYSIS_MODEL_TYPE.split("_")[1]
    weights_path = os.path.join(exp_dir, f"{exp_name}_Epoch_{epoch_num}.pth")
else:
    weights_path = os.path.join(exp_dir, ANALYSIS_MODEL_TYPE)

# =============================================================================
# 설정 출력
# =============================================================================
print(f"\n{'='*80}")
print(f"ECG Model Analysis")
print(f"{'='*80}")
print(f"Device      : {device}")
print(f"Experiment  : {exp_name}")
print(f"Model Type  : {ANALYSIS_MODEL_TYPE}")
print(f"Model Path  : {weights_path}")
print(f"{'='*80}")

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model not found: {weights_path}")

# =============================================================================
# 데이터 로드
# =============================================================================
print("\n[1/3] Loading test data...")

test_data, test_labels, test_rr, test_patient_id, test_sample_id = load_or_extract_data(
    record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS,
    out_len=OUT_LEN, split_name="Test"
)

test_dataset = ECGDataset(test_data, test_rr, test_labels, test_patient_id, test_sample_id)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"  Test samples: {len(test_labels):,}")
print(f"  Distribution: {dict(Counter(test_labels))}")

# =============================================================================
# 모델 로드 (★ get_model로 자동 선택)
# =============================================================================
print("\n[2/3] Loading model...")

all_train_records = DS1_TRAIN + DS1_VALID
model = get_model(
    exp_name=exp_name,
    nOUT=len(CLASSES),
    n_pid=len(all_train_records),
    **MODEL_CONFIG
).to(device)

# 체크포인트 로드
checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"  Loaded from epoch: {checkpoint.get('epoch', 'N/A')}")

# =============================================================================
# 평가
# =============================================================================
print("\n[3/3] Evaluating...")

y_pred, y_true, eval_results = evaluate(model, test_loader, device)
metrics = calculate_metrics(y_true, y_pred)
print_metrics(metrics, CLASSES)

# =============================================================================
# 결과 저장
# =============================================================================
analysis_dir = os.path.join(exp_dir, f"analysis_{ANALYSIS_MODEL_TYPE}")
os.makedirs(analysis_dir, exist_ok=True)

csv_path = os.path.join(analysis_dir, f"results_{ANALYSIS_MODEL_TYPE}.csv")
cm_path = os.path.join(analysis_dir, f"confusion_matrix_{ANALYSIS_MODEL_TYPE}.png")

save_results_csv(metrics, CLASSES, csv_path)
save_confusion_matrix(metrics['confusion_matrix'], CLASSES, cm_path)

# =============================================================================
# Ambiguous 샘플 시각화 (선택)
# =============================================================================
PLOT_AMBIGUOUS = False

if PLOT_AMBIGUOUS:
    print(f"\nSaving ambiguous samples...")
    incorrect_dir = os.path.join(analysis_dir, 'incorrect')
    correct_dir = os.path.join(analysis_dir, 'correct')
    os.makedirs(incorrect_dir, exist_ok=True)
    os.makedirs(correct_dir, exist_ok=True)
    
    plot_ambiguous_sveb(incorrect_dir, eval_results, test_loader, CLASSES, 1, 0, 0.25, 50)
    plot_ambiguous_correct_sveb(correct_dir, eval_results, test_loader, 1, 0.25, 50)
    plot_ambiguous_n_as_s(incorrect_dir, eval_results, test_loader, CLASSES, 1, 0, 0.25, 50)

# =============================================================================
# 완료
# =============================================================================
print(f"\n{'='*80}")
print(f"Analysis Complete!")
print(f"{'='*80}")
print(f"Results: {analysis_dir}")
print(f"{'='*80}")
