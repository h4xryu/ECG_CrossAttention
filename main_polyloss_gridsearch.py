# main_polyloss_gridsearch.py - Poly Loss Grid Search
# main.py 구조 기반, experiments 설정만 추가

import os
import time
from collections import Counter
import torch
from torch.utils.data import DataLoader

from config import (
    DATA_PATH, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, SEED,
    VALID_LEADS, OUT_LEN, CLASSES, DS1_TRAIN, DS1_VALID, DS2_TEST,
    EXP_NAME, MODEL_CONFIG, create_experiment_dir, OUTPUT_PATH
)
from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset
from train import train_one_epoch, validate, save_model
from test import evaluate, calculate_metrics, print_metrics, save_results_csv, save_confusion_matrix
from logger import (
    TrainingLogger, print_epoch_header, print_per_class_metrics,
    print_epoch_stats, print_confidence_stats, print_epoch_time
)

# =============================================================================
# ★ Grid Search 설정 (이것만 수정하면 됨!)
# =============================================================================
experiments = [
    # Experiment 1: Only drop linear term
    {'alpha1': -1.0, 'alpha2': 0.0, 'name': 'drop_linear'},
    
    # Experiment 2: Drop both linear and quadratic terms
    {'alpha1': -1.0, 'alpha2': -0.5, 'name': 'drop_both'},
]

# Experiment 3~N: Poly-1/2 grid search
for a1 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for a2 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        experiments.append({
            'alpha1': a1,
            'alpha2': a2,
            'name': f'poly_a1_{a1}_a2_{a2}'
        })

# =============================================================================
# 초기화 (공통)
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(SEED)

print(f"\n{'='*80}")
print(f"Poly Loss Grid Search")
print(f"{'='*80}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total experiments: {len(experiments)}")
print(f"{'='*80}")

# =============================================================================
# 데이터 로드 (한번만)
# =============================================================================
print("\n[1/2] Loading data...")

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
# Grid Search 결과 저장용
# =============================================================================
all_results = []

# =============================================================================
# [2/2] Grid Search 실험 루프
# =============================================================================
print(f"\n[2/2] Running {len(experiments)} experiments...")

all_train_records = DS1_TRAIN + DS1_VALID
total_grid_start = time.time()

for exp_idx, exp_config in enumerate(experiments, 1):
    ALPHA1 = exp_config['alpha1']
    ALPHA2 = exp_config['alpha2']
    exp_name = f"{EXP_NAME}_{exp_config['name']}"
    
    print(f"\n{'='*80}")
    print(f"[{exp_idx}/{len(experiments)}] {exp_name}")
    print(f"  α₁={ALPHA1}, α₂={ALPHA2}")
    print(f"{'='*80}")
    
    exp_dir = create_experiment_dir(OUTPUT_PATH, exp_name)
    print(f"Output: {exp_dir}")
    
    # ---- 모델 생성 ----
    model = get_model(
        exp_name=EXP_NAME,  # 모델 타입은 config의 EXP_NAME 사용
        nOUT=len(CLASSES),
        n_pid=len(all_train_records),
        **MODEL_CONFIG
    ).to(device)
    
    # ---- 학습 설정 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    logger = TrainingLogger(os.path.join(exp_dir, 'runs'))
    
    best = {
        'auprc': {'value': 0.0, 'epoch': 0, 'path': None},
        'auroc': {'value': 0.0, 'epoch': 0, 'path': None},
        'recall': {'value': 0.0, 'epoch': 0, 'path': None},
    }
    
    # ---- 학습 루프 ----
    print_epoch_header()
    exp_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_metrics, *p_t_train = train_one_epoch(
            model, train_loader, ALPHA1, ALPHA2, optimizer, device
        )
        current_lr = optimizer.param_groups[0]['lr']
        
        print_epoch_stats(epoch, train_loss, train_metrics['acc'], current_lr, phase='Train')
        print_per_class_metrics(train_metrics, CLASSES, phase='Train')
        print_confidence_stats(*p_t_train, phase='Train')
        logger.log_epoch(epoch, train_loss, train_metrics, phase='train')
        logger.log_confidence(epoch, *p_t_train, phase='train')
        
        # Validation
        valid_loss, valid_metrics, *p_t_valid = validate(
            model, valid_loader, ALPHA1, ALPHA2, device
        )
        
        print_epoch_stats(epoch, valid_loss, valid_metrics['acc'], current_lr, phase='Valid')
        print_per_class_metrics(valid_metrics, CLASSES, phase='Valid')
        print_confidence_stats(*p_t_valid, phase='Valid')
        logger.log_epoch(epoch, valid_loss, valid_metrics, phase='valid')
        logger.log_confidence(epoch, *p_t_valid, phase='valid')
        
        # Best model selection
        if valid_metrics['macro_auprc'] > best['auprc']['value']:
            best['auprc'] = {'value': valid_metrics['macro_auprc'], 'epoch': epoch,
                             'path': os.path.join(exp_dir, f'best_model_auprc_{exp_name}.pth')}
            save_model(model, optimizer, epoch, valid_metrics, best['auprc']['path'])
            print(f"  ★ [BEST AUPRC] {best['auprc']['value']:.4f}")
        
        if valid_metrics['macro_auroc'] > best['auroc']['value']:
            best['auroc'] = {'value': valid_metrics['macro_auroc'], 'epoch': epoch,
                             'path': os.path.join(exp_dir, f'best_model_auroc_{exp_name}.pth')}
            save_model(model, optimizer, epoch, valid_metrics, best['auroc']['path'])
            print(f"  ★ [BEST AUROC] {best['auroc']['value']:.4f}")
        
        if valid_metrics['macro_recall'] > best['recall']['value']:
            best['recall'] = {'value': valid_metrics['macro_recall'], 'epoch': epoch,
                              'path': os.path.join(exp_dir, f'best_model_recall_{exp_name}.pth')}
            save_model(model, optimizer, epoch, valid_metrics, best['recall']['path'])
            print(f"  ★ [BEST Recall] {best['recall']['value']:.4f}")
        
        scheduler.step()
        print_epoch_time(epoch, time.time() - epoch_start)
        print("=" * 120 + "\n")
    
    logger.close()
    
    # ---- Test 평가 (Best AUPRC 모델) ----
    if best['auprc']['path'] and os.path.exists(best['auprc']['path']):
        checkpoint = torch.load(best['auprc']['path'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        y_pred, y_true, _ = evaluate(model, test_loader, device)
        test_metrics = calculate_metrics(y_true, y_pred)
        print_metrics(test_metrics, CLASSES)
        
        save_results_csv(test_metrics, CLASSES, os.path.join(exp_dir, f'results_{exp_name}.csv'))
        save_confusion_matrix(test_metrics['confusion_matrix'], CLASSES,
                              os.path.join(exp_dir, f'confusion_matrix_{exp_name}.png'))
    else:
        test_metrics = {'overall_accuracy': 0, 'macro_f1': 0, 'weighted_f1': 0}
    
    # ---- 결과 저장 ----
    result = {
        'exp_name': exp_name,
        'alpha1': ALPHA1,
        'alpha2': ALPHA2,
        'best_valid_auprc': best['auprc']['value'],
        'best_valid_auroc': best['auroc']['value'],
        'best_valid_recall': best['recall']['value'],
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'test_weighted_f1': test_metrics['weighted_f1'],
        'time_min': (time.time() - exp_start) / 60,
    }
    all_results.append(result)
    
    print(f"\n[{exp_idx}/{len(experiments)}] {exp_name} completed in {result['time_min']:.1f} min")
    print(f"  Valid AUPRC: {best['auprc']['value']:.4f}")
    print(f"  Test Acc: {test_metrics['overall_accuracy']:.4f}")

# =============================================================================
# Grid Search 결과 요약
# =============================================================================
print(f"\n{'='*80}")
print(f"GRID SEARCH COMPLETED")
print(f"{'='*80}")
print(f"Total time: {(time.time() - total_grid_start)/60:.1f} min")
print(f"Total experiments: {len(experiments)}")

# 결과 정렬 (Valid AUPRC 기준)
all_results_sorted = sorted(all_results, key=lambda x: x['best_valid_auprc'], reverse=True)

print(f"\nTop 5 Results (by Valid AUPRC):")
print(f"{'Rank':<6} {'Exp Name':<35} {'α1':<6} {'α2':<6} {'V.AUPRC':<10} {'T.Acc':<10}")
print("-" * 80)
for i, r in enumerate(all_results_sorted[:5], 1):
    print(f"{i:<6} {r['exp_name']:<35} {r['alpha1']:<6} {r['alpha2']:<6} "
          f"{r['best_valid_auprc']:<10.4f} {r['test_accuracy']:<10.4f}")

# CSV로 전체 결과 저장
import pandas as pd
results_df = pd.DataFrame(all_results_sorted)
results_csv_path = os.path.join(OUTPUT_PATH, 'gridsearch_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"\nAll results saved to: {results_csv_path}")
print(f"{'='*80}")
