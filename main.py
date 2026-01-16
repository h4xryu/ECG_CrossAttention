# main.py - 학습 스크립트
# 사용법: config.py에서 EXP_NAME 설정 후 python main.py

import os
import time
from collections import Counter
import torch
from torch.utils.data import DataLoader

from config import (
    DATA_PATH, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, SEED, 
    POLY1_EPS, POLY2_EPS, VALID_LEADS, OUT_LEN, CLASSES, 
    DS1_TRAIN, DS1_VALID, DS2_TEST, EXP_NAME, MODEL_CONFIG,
    create_experiment_dir
)
from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset
from train import train_one_epoch, validate, save_model
from test import evaluate, calculate_metrics, print_metrics, save_results_excel, save_confusion_matrix
from logger import (
    TrainingLogger, print_epoch_header, print_per_class_metrics,
    print_epoch_stats, print_confidence_stats, print_epoch_time
)

# =============================================================================
# 초기화
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(SEED)

print(f"\n{'='*80}")
print(f"ECG Classification Training")
print(f"{'='*80}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Experiment: {EXP_NAME}")

exp_dir = create_experiment_dir()

# 가중치 저장 폴더 생성
model_weights_dir = os.path.join(exp_dir, 'model_weights')
best_weights_dir = os.path.join(exp_dir, 'best_weights')
os.makedirs(model_weights_dir, exist_ok=True)
os.makedirs(best_weights_dir, exist_ok=True)

print(f"Output: {exp_dir}")
print(f"  - Model weights: {model_weights_dir}")
print(f"  - Best weights: {best_weights_dir}")
print(f"{'='*80}")

# =============================================================================
# 데이터 로드
# =============================================================================
print("\n[1/3] Loading data...")

train_data, train_labels, train_rr, train_patient_id, train_sample_id = load_or_extract_data(
    record_list=DS1_TRAIN, base_path=DATA_PATH, valid_leads=VALID_LEADS, 
    out_len=OUT_LEN, split_name="Train"
)
valid_data, valid_labels, valid_rr, valid_patient_id, valid_sample_id = load_or_extract_data(
    record_list=DS1_VALID, base_path=DATA_PATH, valid_leads=VALID_LEADS, 
    out_len=OUT_LEN, split_name="Valid"
)
test_data, test_labels, test_rr, test_patient_id, test_sample_id = load_or_extract_data(
    record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS, 
    out_len=OUT_LEN, split_name="Test"
)

# DataLoader 생성
train_dataset = ECGDataset(train_data, train_rr, train_labels, train_patient_id, train_sample_id)
valid_dataset = ECGDataset(valid_data, valid_rr, valid_labels, valid_patient_id, valid_sample_id)
test_dataset = ECGDataset(test_data, test_rr, test_labels, test_patient_id, test_sample_id)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"  Train: {len(train_labels):,} samples | {dict(Counter(train_labels))}")
print(f"  Valid: {len(valid_labels):,} samples | {dict(Counter(valid_labels))}")
print(f"  Test : {len(test_labels):,} samples | {dict(Counter(test_labels))}")

# =============================================================================
# 모델 생성 (get_model로 자동 선택)
# =============================================================================
print("\n[2/3] Creating model...")

all_train_records = DS1_TRAIN + DS1_VALID

model = get_model(
    exp_name=EXP_NAME,
    nOUT=len(CLASSES),
    n_pid=len(all_train_records),
    **MODEL_CONFIG
).to(device)

# =============================================================================
# 학습 설정
# =============================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
logger = TrainingLogger(os.path.join(exp_dir, 'runs'))

# Best model tracking
best = {
    'auprc': {'value': 0.0, 'epoch': 0, 'path': None},
    'auroc': {'value': 0.0, 'epoch': 0, 'path': None},
    'recall': {'value': 0.0, 'epoch': 0, 'path': None},
    'last': {'value': 0.0, 'epoch': 0, 'path': None},
}

# =============================================================================
# 학습 루프
# =============================================================================
print("\n[3/3] Training...")
print_epoch_header()

total_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # ---- Train ----
    train_loss, train_metrics, *p_t_train = train_one_epoch(
        model, train_loader, POLY1_EPS, POLY2_EPS, optimizer, device
    )
    current_lr = optimizer.param_groups[0]['lr']
    
    print_epoch_stats(epoch, train_loss, train_metrics['acc'], current_lr, phase='Train')
    print_per_class_metrics(train_metrics, CLASSES, phase='Train')
    print_confidence_stats(*p_t_train, phase='Train')
    logger.log_epoch(epoch, train_loss, train_metrics, phase='train')
    logger.log_confidence(epoch, *p_t_train, phase='train')
    
    # ---- Validation ----
    valid_loss, valid_metrics, *p_t_valid = validate(
        model, valid_loader, POLY1_EPS, POLY2_EPS, device
    )
    
    print_epoch_stats(epoch, valid_loss, valid_metrics['acc'], current_lr, phase='Valid')
    print_per_class_metrics(valid_metrics, CLASSES, phase='Valid')
    print_confidence_stats(*p_t_valid, phase='Valid')
    logger.log_epoch(epoch, valid_loss, valid_metrics, phase='valid')
    logger.log_confidence(epoch, *p_t_valid, phase='valid')
    
    # 매 에폭 모델 저장 (model_weights 폴더)
    save_model(model, optimizer, epoch, valid_metrics, 
               os.path.join(model_weights_dir, f'{EXP_NAME}_Epoch_{epoch}.pth'))
    
    # ---- Best model selection (best_weights 폴더) ----
    if valid_metrics['macro_auprc'] > best['auprc']['value']:
        best['auprc'] = {'value': valid_metrics['macro_auprc'], 'epoch': epoch,
                         'path': os.path.join(best_weights_dir, f'best_model_auprc_{EXP_NAME}.pth')}
        save_model(model, optimizer, epoch, valid_metrics, best['auprc']['path'])
        print(f"  ★ [BEST AUPRC] {best['auprc']['value']:.4f}")
    
    if valid_metrics['macro_auroc'] > best['auroc']['value']:
        best['auroc'] = {'value': valid_metrics['macro_auroc'], 'epoch': epoch,
                       'path': os.path.join(best_weights_dir, f'best_model_auroc_{EXP_NAME}.pth')}
        save_model(model, optimizer, epoch, valid_metrics, best['auroc']['path'])
        print(f"  ★ [BEST AUROC] {best['auroc']['value']:.4f}")
    
    if valid_metrics['macro_recall'] > best['recall']['value']:
        best['recall'] = {'value': valid_metrics['macro_recall'], 'epoch': epoch,
                          'path': os.path.join(best_weights_dir, f'best_model_recall_{EXP_NAME}.pth')}
        save_model(model, optimizer, epoch, valid_metrics, best['recall']['path'])
        print(f"  ★ [BEST Recall] {best['recall']['value']:.4f}")
    if epoch == EPOCHS:
        best['last'] = {'value': valid_metrics['acc'], 'epoch': epoch,
                        'path': os.path.join(best_weights_dir, f'best_model_last_{EXP_NAME}.pth')}
        save_model(model, optimizer, epoch, valid_metrics, best['last']['path'])
        print(f"  ★ [BEST LAST Epoch] {best['last']['value']:.4f} (epoch {best['last']['epoch']})")
    
    scheduler.step()
    print_epoch_time(epoch, time.time() - epoch_start)
    print("=" * 120 + "\n")

logger.close()

# =============================================================================
# 학습 완료 요약
# =============================================================================
print(f"\n{'='*80}")
print("Training Complete!")
print(f"{'='*80}")
print(f"Total time: {(time.time() - total_start)/60:.1f} min")
print(f"\nBest Models:")
for tag, info in best.items():
    print(f"  {tag.upper():8s}: {info['value']:.4f} (epoch {info['epoch']})")

# =============================================================================
# Test 평가 (Best 모델들)
# =============================================================================
print(f"\n{'='*80}")
print("Test Set Evaluation")
print(f"{'='*80}")

for tag, info in best.items():
    if info['path'] is None or not os.path.exists(info['path']):
        continue
    
    print(f"\n--- Best {tag.upper()} (epoch {info['epoch']}) ---")
    
    checkpoint = torch.load(info['path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    y_pred, y_true, _ = evaluate(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, CLASSES)
    
    # 결과 저장 (Excel 4개 시트: Macro, Weighted, Per_Class, Confusion_Matrix)
    save_results_excel(metrics, CLASSES, os.path.join(exp_dir, f'results_best_{tag}_{EXP_NAME}.xlsx'))
    save_confusion_matrix(metrics['confusion_matrix'], CLASSES, 
                          os.path.join(exp_dir, f'confusion_matrix_best_{tag}_{EXP_NAME}.png'))

print(f"\n{'='*80}")
print(f"All results saved to: {exp_dir}")
print(f"{'='*80}")
