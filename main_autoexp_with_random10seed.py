# main_autoexp_with_random10seed.py - 10개 시드 자동 실험 스크립트
# 각 실험을 10개의 다른 시드로 수행하여 mean ± std 계산
#
# 실험 종류:
#   A0* (Baseline), A1* (Naive Concat), A2* (Cross Attention) - Dense Block 포함
#   B0*, B1*, B2* - Dense Block 없음
#   * : DS1 전체 22명 train (50 epoch 고정)
#   @ : DS1-1(17)/DS1-2(5) split, best AUROC epoch
#
# 목적: Naive Concat vs Cross Attention 통계적 비교 검증
#
# 사용법: python main_autoexp_with_random10seed.py

import os
import time
import copy
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset
from train import train_one_epoch, validate
from test import evaluate, calculate_metrics, print_metrics

# =============================================================================
# 실험 설정
# =============================================================================

# 공통 설정
DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH = './auto_results_10seeds/'
BATCH_SIZE = 1024
EPOCHS = 50
LR = 0.0001
WEIGHT_DECAY = 1e-3
POLY1_EPS = 0.0
POLY2_EPS = 0.0
CLASSES = ['N', 'S', 'V', 'F']

# 10개 시드 (재현 가능한 랜덤 시드)
SEEDS = [42, 123, 234, 345, 456, 567, 678, 789, 890, 901]

# RR Feature 설정
RR_FEATURE_OPTION = "opt4"  # numerically stable features
RR_FEATURE_DIMS = {"opt1": 7, "opt2": 38, "opt3": 7, "opt4": 7}

# 모델 설정
MODEL_CONFIG = {
    'in_channels': 1,
    'out_ch': 180,
    'mid_ch': 30,
    'num_heads': 9,
    'n_rr': RR_FEATURE_DIMS[RR_FEATURE_OPTION],
}

# ECG Parameters
VALID_LEADS = ['MLII', 'V1', 'V2', 'V4', 'V5']
OUT_LEN = 720

# =============================================================================
# 데이터 분할 설정
# =============================================================================

# DS1 전체 (22명) - * 실험용
DS1_FULL = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119',
    '122', '201', '203', '209', '215', '223', '230', '208',
    '114', '124', '205', '207', '220'
]

# DS1-1 Train (17명), DS1-2 Valid (5명) - @ 실험용
DS1_TRAIN_SPLIT = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119',
    '122', '201', '203', '209', '215', '223', '230', '208'
]
DS1_VALID_SPLIT = ['114', '124', '205', '207', '220']

# DS2 Test (22명)
DS2_TEST = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]

# =============================================================================
# 실험 정의
# =============================================================================

# 실험 목록: (실험명, 모델타입, 데이터설정)
# 데이터설정: 'star' = DS1 전체, 'at' = DS1-1/DS1-2 split
EXPERIMENTS = [
    # A 시리즈 (Dense Block 포함) - * 설정
    ('A0*', 'baseline', 'star'),
    ('A1*', 'naive_concatenate', 'star'),
    ('A2*', 'cross_attention', 'star'),
    
    # B 시리즈 (Dense Block 없음) - * 설정
    ('B0*', 'baseline_B', 'star'),
    ('B1*', 'naive_concatenate_B', 'star'),
    ('B2*', 'cross_attention_B', 'star'),
    
    # A 시리즈 - @ 설정 (DS1-1/DS1-2 split)
    ('A0@', 'baseline', 'at'),
    ('A1@', 'naive_concatenate', 'at'),
    ('A2@', 'cross_attention', 'at'),
    
    # B 시리즈 - @ 설정
    ('B0@', 'baseline_B', 'at'),
    ('B1@', 'naive_concatenate_B', 'at'),
    ('B2@', 'cross_attention_B', 'at'),
]

# =============================================================================
# 핵심 메트릭 목록
# =============================================================================

METRIC_KEYS = [
    'overall_accuracy',
    'macro_accuracy', 'macro_recall', 'macro_specificity', 'macro_prec', 'macro_f1',
    'weighted_accuracy', 'weighted_recall', 'weighted_specificity', 'weighted_prec', 'weighted_f1',
]

PER_CLASS_METRIC_KEYS = [
    'per_class_accuracy', 'per_class_recall', 'per_class_specificity', 'per_class_precision', 'per_class_f1'
]

# =============================================================================
# 유틸리티 함수
# =============================================================================

def run_single_experiment(exp_name, model_type, data_config, seed, device, 
                          train_data, train_labels, train_rr, train_pid, train_sid,
                          valid_data, valid_labels, valid_rr, valid_pid, valid_sid,
                          test_data, test_labels, test_rr, test_pid, test_sid,
                          n_records):
    """
    단일 시드로 단일 실험 수행
    
    Returns:
        metrics: 테스트 결과 딕셔너리
    """
    set_seed(seed)
    
    # DataLoader 생성
    train_dataset = ECGDataset(train_data, train_rr, train_labels, train_pid, train_sid)
    valid_dataset = ECGDataset(valid_data, valid_rr, valid_labels, valid_pid, valid_sid)
    test_dataset = ECGDataset(test_data, test_rr, test_labels, test_pid, test_sid)
    
    # DataLoader (재현성을 위해 worker_init_fn 추가)
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
    
    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,
                              generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    # 모델 생성
    model = get_model(
        exp_name=model_type,
        nOUT=len(CLASSES),
        n_pid=n_records,
        **MODEL_CONFIG
    ).to(device)
    
    # 학습 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Best 모델 추적 (AUROC 기준) - @ 실험용
    best_auroc = {'value': 0.0, 'epoch': 0, 'state_dict': None}
    last_state_dict = None  # * 실험용 (마지막 에폭)
    
    # 학습 루프
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_metrics, *_ = train_one_epoch(
            model, train_loader, POLY1_EPS, POLY2_EPS, optimizer, device
        )
        
        # Validation (@ 실험에서만 의미 있음)
        if data_config == 'at':
            valid_loss, valid_metrics, *_ = validate(
                model, valid_loader, POLY1_EPS, POLY2_EPS, device
            )
            
            # Best AUROC 체크
            if valid_metrics['macro_auroc'] > best_auroc['value']:
                best_auroc = {
                    'value': valid_metrics['macro_auroc'],
                    'epoch': epoch,
                    'state_dict': copy.deepcopy(model.state_dict())
                }
        
        scheduler.step()
        
        # 마지막 에폭 state_dict 저장 (* 실험용)
        if epoch == EPOCHS:
            last_state_dict = copy.deepcopy(model.state_dict())
    
    # 모델 선택
    if data_config == 'star':
        # * 실험: 마지막 에폭 (50 에폭) weight 사용
        final_state_dict = last_state_dict
    else:
        # @ 실험: best AUROC epoch weight 사용
        final_state_dict = best_auroc['state_dict']
    
    # 선택된 모델로 테스트
    model.load_state_dict(final_state_dict)
    model.eval()
    
    y_pred, y_true, _ = evaluate(model, test_loader, device)
    metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
    
    return metrics


def calculate_mean_std(results_list):
    """
    여러 실험 결과의 mean과 std 계산
    
    Args:
        results_list: list of metrics dictionaries
    
    Returns:
        mean_dict, std_dict
    """
    mean_dict = {}
    std_dict = {}
    
    # 스칼라 메트릭
    for key in METRIC_KEYS:
        values = [r[key] for r in results_list if key in r]
        if values:
            mean_dict[key] = np.mean(values)
            std_dict[key] = np.std(values)
    
    # Per-class 메트릭
    for key in PER_CLASS_METRIC_KEYS:
        values = [r[key] for r in results_list if key in r]
        if values:
            values_arr = np.array(values)  # (n_seeds, n_classes)
            mean_dict[key] = np.mean(values_arr, axis=0)
            std_dict[key] = np.std(values_arr, axis=0)
    
    return mean_dict, std_dict


def format_mean_std(mean, std, precision=4):
    """mean ± std 형식으로 포맷"""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


# =============================================================================
# 결과 저장 함수
# =============================================================================

def save_summary_excel(all_results, output_path):
    """
    모든 실험 결과를 엑셀로 저장
    
    Args:
        all_results: {exp_name: {'mean': mean_dict, 'std': std_dict, 'raw': [metrics_list]}}
        output_path: 출력 파일 경로
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Sheet 1: Macro Metrics Summary
        macro_data = []
        for exp_name in [e[0] for e in EXPERIMENTS]:
            if exp_name not in all_results:
                continue
            mean = all_results[exp_name]['mean']
            std = all_results[exp_name]['std']
            macro_data.append({
                'Experiment': exp_name,
                'Accuracy': format_mean_std(mean['macro_accuracy'], std['macro_accuracy']),
                'Sensitivity': format_mean_std(mean['macro_recall'], std['macro_recall']),
                'Specificity': format_mean_std(mean['macro_specificity'], std['macro_specificity']),
                'Precision': format_mean_std(mean['macro_prec'], std['macro_prec']),
                'F1': format_mean_std(mean['macro_f1'], std['macro_f1']),
            })
        df_macro = pd.DataFrame(macro_data)
        df_macro.to_excel(writer, sheet_name='Macro', index=False)
        
        # Sheet 2: Weighted Metrics Summary
        weighted_data = []
        for exp_name in [e[0] for e in EXPERIMENTS]:
            if exp_name not in all_results:
                continue
            mean = all_results[exp_name]['mean']
            std = all_results[exp_name]['std']
            weighted_data.append({
                'Experiment': exp_name,
                'Accuracy': format_mean_std(mean['weighted_accuracy'], std['weighted_accuracy']),
                'Sensitivity': format_mean_std(mean['weighted_recall'], std['weighted_recall']),
                'Specificity': format_mean_std(mean['weighted_specificity'], std['weighted_specificity']),
                'Precision': format_mean_std(mean['weighted_prec'], std['weighted_prec']),
                'F1': format_mean_std(mean['weighted_f1'], std['weighted_f1']),
            })
        df_weighted = pd.DataFrame(weighted_data)
        df_weighted.to_excel(writer, sheet_name='Weighted', index=False)
        
        # Sheet 3: Per-Class Metrics
        per_class_data = []
        for exp_name in [e[0] for e in EXPERIMENTS]:
            if exp_name not in all_results:
                continue
            mean = all_results[exp_name]['mean']
            std = all_results[exp_name]['std']
            for i, cls in enumerate(CLASSES):
                per_class_data.append({
                    'Experiment': exp_name,
                    'Class': cls,
                    'Accuracy': format_mean_std(mean['per_class_accuracy'][i], std['per_class_accuracy'][i]),
                    'Sensitivity': format_mean_std(mean['per_class_recall'][i], std['per_class_recall'][i]),
                    'Specificity': format_mean_std(mean['per_class_specificity'][i], std['per_class_specificity'][i]),
                    'Precision': format_mean_std(mean['per_class_precision'][i], std['per_class_precision'][i]),
                    'F1': format_mean_std(mean['per_class_f1'][i], std['per_class_f1'][i]),
                })
        df_per_class = pd.DataFrame(per_class_data)
        df_per_class.to_excel(writer, sheet_name='Per-Class', index=False)
        
        # Sheet 4: Raw Data (모든 시드 결과)
        raw_data = []
        for exp_name in [e[0] for e in EXPERIMENTS]:
            if exp_name not in all_results:
                continue
            for seed_idx, metrics in enumerate(all_results[exp_name]['raw']):
                raw_data.append({
                    'Experiment': exp_name,
                    'Seed': SEEDS[seed_idx],
                    'Overall_Acc': metrics['overall_accuracy'],
                    'Macro_Acc': metrics['macro_accuracy'],
                    'Macro_Recall': metrics['macro_recall'],
                    'Macro_Spec': metrics['macro_specificity'],
                    'Macro_Prec': metrics['macro_prec'],
                    'Macro_F1': metrics['macro_f1'],
                })
        df_raw = pd.DataFrame(raw_data)
        df_raw.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        # Sheet 5: Comparison (Naive vs Cross-Attention)
        comparison_data = []
        comparisons = [
            ('A1* vs A2*', 'A1*', 'A2*'),  # Dense, star
            ('B1* vs B2*', 'B1*', 'B2*'),  # No Dense, star
            ('A1@ vs A2@', 'A1@', 'A2@'),  # Dense, at
            ('B1@ vs B2@', 'B1@', 'B2@'),  # No Dense, at
        ]
        
        for comp_name, naive_exp, cross_exp in comparisons:
            if naive_exp not in all_results or cross_exp not in all_results:
                continue
            
            naive_f1 = [r['macro_f1'] for r in all_results[naive_exp]['raw']]
            cross_f1 = [r['macro_f1'] for r in all_results[cross_exp]['raw']]
            
            naive_acc = [r['overall_accuracy'] for r in all_results[naive_exp]['raw']]
            cross_acc = [r['overall_accuracy'] for r in all_results[cross_exp]['raw']]
            
            # t-test
            try:
                from scipy import stats
                t_stat_f1, p_value_f1 = stats.ttest_ind(naive_f1, cross_f1)
                t_stat_acc, p_value_acc = stats.ttest_ind(naive_acc, cross_acc)
            except ImportError:
                p_value_f1 = p_value_acc = float('nan')
            
            comparison_data.append({
                'Comparison': comp_name,
                'Naive_F1_mean': np.mean(naive_f1),
                'Naive_F1_std': np.std(naive_f1),
                'Cross_F1_mean': np.mean(cross_f1),
                'Cross_F1_std': np.std(cross_f1),
                'F1_Diff': np.mean(cross_f1) - np.mean(naive_f1),
                'F1_p-value': p_value_f1,
                'Naive_Acc_mean': np.mean(naive_acc),
                'Naive_Acc_std': np.std(naive_acc),
                'Cross_Acc_mean': np.mean(cross_acc),
                'Cross_Acc_std': np.std(cross_acc),
                'Acc_Diff': np.mean(cross_acc) - np.mean(naive_acc),
                'Acc_p-value': p_value_acc,
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_excel(writer, sheet_name='Comparison', index=False)
    
    print(f"\n✅ Summary saved to: {output_path}")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("ECG Cross-Attention Multi-Seed Experiment (10 seeds)")
    print("="*80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Experiments: {len(EXPERIMENTS)} x {len(SEEDS)} seeds = {len(EXPERIMENTS) * len(SEEDS)} runs")
    print(f"Seeds: {SEEDS}")
    print("="*80)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 데이터 미리 로드 (시드별로 데이터는 동일하므로 한 번만 로드)
    print("\n📂 Loading data...")
    
    # DS1 전체 (* 실험용)
    train_data_star, train_labels_star, train_rr_star, train_pid_star, train_sid_star = load_or_extract_data(
        record_list=DS1_FULL, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name="Train_Star"
    )
    n_records_star = len(DS1_FULL)
    
    # DS1-1 Train (@ 실험용)
    train_data_at, train_labels_at, train_rr_at, train_pid_at, train_sid_at = load_or_extract_data(
        record_list=DS1_TRAIN_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name="Train_At"
    )
    
    # DS1-2 Valid (@ 실험용)
    valid_data_at, valid_labels_at, valid_rr_at, valid_pid_at, valid_sid_at = load_or_extract_data(
        record_list=DS1_VALID_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name="Valid_At"
    )
    n_records_at = len(DS1_TRAIN_SPLIT) + len(DS1_VALID_SPLIT)
    
    # DS2 Test (모든 실험 공통)
    test_data, test_labels, test_rr, test_pid, test_sid = load_or_extract_data(
        record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name="Test"
    )
    
    print(f"  Star Train: {len(train_labels_star):,} samples")
    print(f"  At Train: {len(train_labels_at):,} samples")
    print(f"  At Valid: {len(valid_labels_at):,} samples")
    print(f"  Test: {len(test_labels):,} samples")
    
    # 결과 저장용
    all_results = {}
    
    total_start = time.time()
    
    # 모든 실험 수행
    for exp_idx, (exp_name, model_type, data_config) in enumerate(EXPERIMENTS):
        print(f"\n{'='*80}")
        print(f"[{exp_idx+1}/{len(EXPERIMENTS)}] Experiment: {exp_name}")
        print(f"Model: {model_type}, Data: {data_config}")
        print(f"{'='*80}")
        
        # 데이터 선택
        if data_config == 'star':
            train_data = train_data_star
            train_labels = train_labels_star
            train_rr = train_rr_star
            train_pid = train_pid_star
            train_sid = train_sid_star
            valid_data = train_data_star  # star는 validation 없음
            valid_labels = train_labels_star
            valid_rr = train_rr_star
            valid_pid = train_pid_star
            valid_sid = train_sid_star
            n_records = n_records_star
        else:
            train_data = train_data_at
            train_labels = train_labels_at
            train_rr = train_rr_at
            train_pid = train_pid_at
            train_sid = train_sid_at
            valid_data = valid_data_at
            valid_labels = valid_labels_at
            valid_rr = valid_rr_at
            valid_pid = valid_pid_at
            valid_sid = valid_sid_at
            n_records = n_records_at
        
        # 10개 시드로 실험
        seed_results = []
        
        for seed_idx, seed in enumerate(SEEDS):
            exp_start = time.time()
            print(f"\n  Seed {seed_idx+1}/{len(SEEDS)} (seed={seed})...", end=' ')
            
            try:
                metrics = run_single_experiment(
                    exp_name, model_type, data_config, seed, device,
                    train_data, train_labels, train_rr, train_pid, train_sid,
                    valid_data, valid_labels, valid_rr, valid_pid, valid_sid,
                    test_data, test_labels, test_rr, test_pid, test_sid,
                    n_records
                )
                seed_results.append(metrics)
                exp_time = time.time() - exp_start
                print(f"F1={metrics['macro_f1']:.4f}, Acc={metrics['overall_accuracy']:.4f} ({exp_time/60:.1f}min)")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if seed_results:
            mean_dict, std_dict = calculate_mean_std(seed_results)
            all_results[exp_name] = {
                'mean': mean_dict,
                'std': std_dict,
                'raw': seed_results
            }
            
            # 중간 결과 출력
            print(f"\n  📊 {exp_name} Results (n={len(seed_results)}):")
            print(f"     Macro F1: {format_mean_std(mean_dict['macro_f1'], std_dict['macro_f1'])}")
            print(f"     Overall Acc: {format_mean_std(mean_dict['overall_accuracy'], std_dict['overall_accuracy'])}")
    
    total_time = (time.time() - total_start) / 60
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_xlsx = os.path.join(OUTPUT_PATH, f'MultiSeed_Results_{timestamp}.xlsx')
    save_summary_excel(all_results, output_xlsx)
    
    # 최종 요약 출력
    print("\n" + "="*80)
    print("All Experiments Complete!")
    print("="*80)
    print(f"Total time: {total_time:.1f} min ({total_time/60:.1f} hours)")
    print(f"Total runs: {len(EXPERIMENTS) * len(SEEDS)}")
    
    print(f"\n📊 Summary (Macro F1):")
    print("-"*60)
    for exp_name in [e[0] for e in EXPERIMENTS]:
        if exp_name in all_results:
            mean = all_results[exp_name]['mean']
            std = all_results[exp_name]['std']
            print(f"  {exp_name:6s}: {format_mean_std(mean['macro_f1'], std['macro_f1'])}")
    
    # Naive vs Cross-Attention 비교
    print(f"\n📊 Naive Concat vs Cross-Attention Comparison:")
    print("-"*60)
    comparisons = [
        ('A1* vs A2*', 'A1*', 'A2*'),
        ('B1* vs B2*', 'B1*', 'B2*'),
        ('A1@ vs A2@', 'A1@', 'A2@'),
        ('B1@ vs B2@', 'B1@', 'B2@'),
    ]
    
    for comp_name, naive_exp, cross_exp in comparisons:
        if naive_exp in all_results and cross_exp in all_results:
            naive_f1 = all_results[naive_exp]['mean']['macro_f1']
            cross_f1 = all_results[cross_exp]['mean']['macro_f1']
            diff = cross_f1 - naive_f1
            better = "Cross ✓" if diff > 0 else "Naive ✗"
            print(f"  {comp_name:15s}: diff={diff:+.4f} ({better})")
    
    print(f"\n✅ All results saved to: {output_xlsx}")
    print("="*80)

