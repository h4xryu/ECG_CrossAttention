# main_autoexp.py - 자동 실험 스크립트
# 엑셀 양식의 모든 실험을 자동으로 수행하고 결과를 채움
#
# 실험 종류:
#   A0* (Baseline), A1* (Naive Concat), A2* (Cross Attention) - Dense Block 포함
#   B0*, B1*, B2* - Dense Block 없음
#   * : DS1 전체 22명 train (no validation split)
#   @ : DS1-1(17)/DS1-2(5) split, best AUROC epoch
#
# 사용법: python main_autoexp.py

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
from train import train_one_epoch, validate, save_model
from test import evaluate, calculate_metrics, print_metrics, save_results_excel, save_confusion_matrix
from logger import TrainingLogger

# =============================================================================
# 실험 설정
# =============================================================================

# 공통 설정
DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH = './auto_results/'
BATCH_SIZE = 1024
EPOCHS = 50
LR = 0.0001
WEIGHT_DECAY = 1e-3
SEED = 1234
POLY1_EPS = 0.0
POLY2_EPS = 0.0
CLASSES = ['N', 'S', 'V', 'F']

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
    
    # @ 설정 (DS1-1/DS1-2 split)
    ('A1@', 'naive_concatenate', 'at'),
    ('A2@', 'cross_attention', 'at'),
    ('A0@', 'baseline', 'at'),
    ('B1@', 'naive_concatenate_B', 'at'),
    ('B0@', 'baseline_B', 'at'),
    ('B2@', 'cross_attention_B', 'at'),
]

# =============================================================================
# 유틸리티 함수
# =============================================================================

def create_auto_exp_dir(exp_name):
    """자동 실험용 폴더 생성"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_PATH, f'{exp_name}_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'model_weights'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'best_weights'), exist_ok=True)
    return exp_dir


def run_experiment(exp_name, model_type, data_config, device):
    """
    단일 실험 수행
    
    Args:
        exp_name: 실험 이름 (A0*, A1@, 등)
        model_type: 모델 타입 (baseline, naive_concatenate, cross_attention, *_B)
        data_config: 'star' (DS1 전체) 또는 'at' (DS1-1/DS1-2 split)
        device: torch device
    
    Returns:
        metrics: 테스트 결과 딕셔너리
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"Model: {model_type}, Data: {data_config}")
    print(f"{'='*80}")
    
    set_seed(SEED)
    exp_dir = create_auto_exp_dir(exp_name)
    
    # 데이터 로드
    if data_config == 'star':
        # DS1 전체를 train으로 사용 (validation 없음)
        train_data, train_labels, train_rr, train_pid, train_sid = load_or_extract_data(
            record_list=DS1_FULL, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Train_{exp_name}"
        )
        valid_data, valid_labels, valid_rr, valid_pid, valid_sid = \
            train_data, train_labels, train_rr, train_pid, train_sid
        n_records = len(DS1_FULL)
    else:  # 'at'
        # DS1-1 train, DS1-2 validation
        train_data, train_labels, train_rr, train_pid, train_sid = load_or_extract_data(
            record_list=DS1_TRAIN_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Train_{exp_name}"
        )
        valid_data, valid_labels, valid_rr, valid_pid, valid_sid = load_or_extract_data(
            record_list=DS1_VALID_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Valid_{exp_name}"
        )
        n_records = len(DS1_TRAIN_SPLIT) + len(DS1_VALID_SPLIT)
    
    # Test 데이터
    test_data, test_labels, test_rr, test_pid, test_sid = load_or_extract_data(
        record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name=f"Test_{exp_name}"
    )
    
    # DataLoader 생성
    train_dataset = ECGDataset(train_data, train_rr, train_labels, train_pid, train_sid)
    valid_dataset = ECGDataset(valid_data, valid_rr, valid_labels, valid_pid, valid_sid)
    test_dataset = ECGDataset(test_data, test_rr, test_labels, test_pid, test_sid)
    
    # DataLoader (재현성을 위해 worker_init_fn 추가)
    def worker_init_fn(worker_id):
        np.random.seed(SEED + worker_id)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    print(f"  Train: {len(train_labels):,} samples | {dict(Counter(train_labels))}")
    print(f"  Valid: {len(valid_labels):,} samples | {dict(Counter(valid_labels))}")
    print(f"  Test : {len(test_labels):,} samples | {dict(Counter(test_labels))}")
    
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
                print(f"  Epoch {epoch}: New best AUROC = {best_auroc['value']:.4f}")
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}, "
                      f"Valid AUROC: {valid_metrics['macro_auroc']:.4f}")
        else:
            # * 실험: validation 없이 학습만
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_metrics['acc']:.4f}")
        
        scheduler.step()
        
        # 마지막 에폭 state_dict 저장 (* 실험용)
        if epoch == EPOCHS:
            last_state_dict = copy.deepcopy(model.state_dict())
    
    # 모델 선택 및 저장
    if data_config == 'star':
        # * 실험: 마지막 에폭 (50 에폭) weight 사용
        final_state_dict = last_state_dict
        final_epoch = EPOCHS
        save_path = os.path.join(exp_dir, 'best_weights', f'last_epoch_{exp_name}.pth')
        torch.save({
            'model_state_dict': final_state_dict,
            'epoch': final_epoch,
        }, save_path)
        print(f"\n  Last epoch model saved (epoch {final_epoch})")
    else:
        # @ 실험: best AUROC epoch weight 사용
        final_state_dict = best_auroc['state_dict']
        final_epoch = best_auroc['epoch']
        save_path = os.path.join(exp_dir, 'best_weights', f'best_auroc_{exp_name}.pth')
        torch.save({
            'model_state_dict': final_state_dict,
            'epoch': final_epoch,
            'auroc': best_auroc['value']
        }, save_path)
        print(f"\n  Best AUROC model saved (epoch {final_epoch}, AUROC {best_auroc['value']:.4f})")
    
    # 선택된 모델로 테스트
    model.load_state_dict(final_state_dict)
    model.eval()
    
    y_pred, y_true, _ = evaluate(model, test_loader, device)
    metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
    
    print_metrics(metrics, CLASSES)
    
    # 결과 저장
    save_results_excel(metrics, CLASSES, os.path.join(exp_dir, f'results_{exp_name}.xlsx'))
    save_confusion_matrix(metrics['confusion_matrix'], CLASSES,
                          os.path.join(exp_dir, f'confusion_matrix_{exp_name}.png'))
    
    return metrics, exp_dir


def fill_excel_template(all_results, template_path, output_path):
    """
    엑셀 양식에 결과 채우기
    
    Args:
        all_results: {exp_name: metrics} 딕셔너리
        template_path: 양식 파일 경로
        output_path: 출력 파일 경로
    """
    # 양식 읽기
    df = pd.read_excel(template_path, sheet_name='Performance Metrics', header=None)
    
    # 실험명 -> 행 번호 매핑
    exp_row_map = {
        'A0*': 3, 'A1*': 4, 'A2*': 5,
        'B0*': 6, 'B1*': 7, 'B2*': 8,
        'A1@': 9, 'B2@': 10
    }
    
    # 열 매핑 (0-indexed)
    # Macro: Acc(1), Sens(2), Spec(3), Prec(4), F1(5)
    # Weighted: Acc(6), Sens(7), Spec(8), Prec(9), F1(10)
    # N: Acc(11), Sens(12), Spec(13), Prec(14), F1(15)
    # S: Acc(16), Sens(17), Spec(18), Prec(19), F1(20)
    # V: Acc(21), Sens(22), Spec(23), Prec(24), F1(25)
    # F: Acc(26), Sens(27), Spec(28), Prec(29), F1(30)
    
    for exp_name, metrics in all_results.items():
        if exp_name not in exp_row_map:
            continue
        
        row = exp_row_map[exp_name]
        
        # Macro metrics
        df.iloc[row, 1] = metrics['macro_accuracy']
        df.iloc[row, 2] = metrics['macro_recall']  # Sensitivity
        df.iloc[row, 3] = metrics['macro_specificity']
        df.iloc[row, 4] = metrics['macro_prec']
        df.iloc[row, 5] = metrics['macro_f1']
        
        # Weighted metrics
        df.iloc[row, 6] = metrics['weighted_accuracy']
        df.iloc[row, 7] = metrics['weighted_recall']
        df.iloc[row, 8] = metrics['weighted_specificity']
        df.iloc[row, 9] = metrics['weighted_prec']
        df.iloc[row, 10] = metrics['weighted_f1']
        
        # Per-class metrics (N, S, V, F)
        for i, cls in enumerate(CLASSES):
            base_col = 11 + i * 5
            df.iloc[row, base_col] = metrics['per_class_accuracy'][i]
            df.iloc[row, base_col + 1] = metrics['per_class_recall'][i]
            df.iloc[row, base_col + 2] = metrics['per_class_specificity'][i]
            df.iloc[row, base_col + 3] = metrics['per_class_precision'][i]
            df.iloc[row, base_col + 4] = metrics['per_class_f1'][i]
    
    # 저장
    df.to_excel(output_path, sheet_name='Performance Metrics', index=False, header=False)
    print(f"\n✅ Results saved to: {output_path}")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("ECG Cross-Attention Auto Experiment")
    print("="*80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print("="*80)
    
    # 결과 저장용
    all_results = {}
    all_exp_dirs = {}
    
    total_start = time.time()
    
    # 모든 실험 수행
    for exp_name, model_type, data_config in EXPERIMENTS:
        try:
            metrics, exp_dir = run_experiment(exp_name, model_type, data_config, device)
            all_results[exp_name] = metrics
            all_exp_dirs[exp_name] = exp_dir
        except Exception as e:
            print(f"\n❌ Error in {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = (time.time() - total_start) / 60
    
    # 엑셀 양식에 결과 채우기
    template_path = 'Macro_및_Per-class_분류_성능_지표_엑셀_양식-Genspark_AI_Sheets-20260119_1022.xlsx'
    if os.path.exists(template_path):
        output_xlsx = os.path.join(OUTPUT_PATH, f'Results_Summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        fill_excel_template(all_results, template_path, output_xlsx)
    
    # 최종 요약 출력
    print("\n" + "="*80)
    print("All Experiments Complete!")
    print("="*80)
    print(f"Total time: {total_time:.1f} min")
    print(f"\nResults:")
    
    for exp_name in all_results:
        m = all_results[exp_name]
        print(f"  {exp_name:6s}: Macro F1={m['macro_f1']:.4f}, "
              f"Weighted F1={m['weighted_f1']:.4f}, "
              f"Acc={m['overall_accuracy']:.4f}")
    
    print(f"\nAll results saved to: {OUTPUT_PATH}")
    print("="*80)

