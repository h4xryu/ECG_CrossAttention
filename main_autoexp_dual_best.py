# main_autoexp_dual_best.py - Best AUROC & Best AUPRC 동시 저장 실험 스크립트
# 각 에폭에서 validation AUROC, AUPRC를 계산하고 각각의 best epoch 모델 저장
#
# 사용법: python main_autoexp_dual_best.py
#
# 출력:
#   - best_auroc_*.pth: Best AUROC epoch 모델
#   - best_auprc_*.pth: Best AUPRC epoch 모델
#   - New_Analysis_Results_*.xlsx: 결과 엑셀 (템플릿 형식)

import os
import time
import copy
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from config import (
    DATA_PATH, BATCH_SIZE, SEED, VALID_LEADS, OUT_LEN, CLASSES,
    RR_FEATURE_OPTION, RR_FEATURE_DIMS, MODEL_CONFIG,
    DS1_FULL, DS1_TRAIN, DS1_VALID, DS2_TEST,
    EPOCHS, LR, WEIGHT_DECAY, POLY1_EPS, POLY2_EPS
)
from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset
from train import train_one_epoch, validate

# =============================================================================
# 설정
# =============================================================================

OUTPUT_PATH = './dual_best_results/'

# 실험 목록 (엑셀 양식 기준)
EXPERIMENTS = [
    # * 설정 (DS1 전체, 50 epoch 고정)
    ('A0*', 'baseline', 'star'),
    ('A1*', 'naive_concatenate', 'star'),
    ('A2*', 'cross_attention', 'star'),
    ('B0*', 'baseline_B', 'star'),
    ('B1*', 'naive_concatenate_B', 'star'),
    ('B2*', 'cross_attention_B', 'star'),
    
    # @ 설정 (DS1-1/DS1-2 split, best epoch)
    ('A0@', 'baseline', 'at'),
    ('A1@', 'naive_concatenate', 'at'),
    ('A2@', 'cross_attention', 'at'),
    ('B1@', 'naive_concatenate_B', 'at'),
    ('B2@', 'cross_attention_B', 'at'),
]

# 실험명 -> 행 번호 매핑 (엑셀 템플릿 기준)
EXP_ROW_MAP = {
    'A0*': 2, 'A0@': 3,
    'A1*': 4, 'A1@': 5,
    'A2*': 6, 'A2@': 7,
    'B0*': 8, 'B0@': 9,
    'B1*': 10, 'B1@': 11,
    'B2*': 12, 'B2@': 13,
}

# =============================================================================
# 평가 함수
# =============================================================================

def evaluate_with_auc(model, data_loader, device, classes):
    """AUROC, AUPRC 포함 전체 평가"""
    model.eval()
    y_pred, y_true, y_probs = [], [], []
    
    with torch.no_grad():
        for batch in data_loader:
            ecg_inputs, rr_features, labels, patient_id, idx = batch
            ecg_inputs = ecg_inputs.to(device)
            rr_features = rr_features.to(device)
            
            logits, _ = model(ecg_inputs, rr_features)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.numpy())
            y_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    n_classes = len(classes)
    
    # One-hot
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1
    
    # Per-class AUROC, AUPRC
    per_class_auroc = []
    per_class_auprc = []
    
    for i in range(n_classes):
        try:
            auroc = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
        except:
            auroc = 0.0
        try:
            auprc = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        except:
            auprc = 0.0
        per_class_auroc.append(auroc)
        per_class_auprc.append(auprc)
    
    # Aggregated
    accuracy = accuracy_score(y_true, y_pred)
    macro_auroc = np.mean(per_class_auroc)
    macro_auprc = np.mean(per_class_auprc)
    
    class_counts = np.bincount(y_true, minlength=n_classes)
    class_weights = class_counts / len(y_true)
    weighted_auroc = np.sum(np.array(per_class_auroc) * class_weights)
    weighted_auprc = np.sum(np.array(per_class_auprc) * class_weights)
    
    return {
        'accuracy': accuracy,
        'macro_auroc': macro_auroc,
        'macro_auprc': macro_auprc,
        'weighted_auroc': weighted_auroc,
        'weighted_auprc': weighted_auprc,
        'per_class_auroc': per_class_auroc,
        'per_class_auprc': per_class_auprc,
    }


def run_experiment(exp_name, model_type, data_config, device):
    """단일 실험 수행 - Best AUROC & Best AUPRC 모두 저장"""
    
    set_seed(SEED)
    
    # 실험 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_PATH, f'{exp_name}_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'best_weights'), exist_ok=True)
    
    # 데이터 로드
    if data_config == 'star':
        train_records = DS1_FULL
        valid_records = DS1_FULL  # star는 validation 없음
        n_records = len(DS1_FULL)
    else:  # 'at'
        train_records = DS1_TRAIN
        valid_records = DS1_VALID
        n_records = len(DS1_TRAIN) + len(DS1_VALID)
    
    train_data, train_labels, train_rr, train_pid, train_sid = load_or_extract_data(
        record_list=train_records, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name=f"Train_{exp_name}"
    )
    valid_data, valid_labels, valid_rr, valid_pid, valid_sid = load_or_extract_data(
        record_list=valid_records, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name=f"Valid_{exp_name}"
    )
    test_data, test_labels, test_rr, test_pid, test_sid = load_or_extract_data(
        record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name=f"Test_{exp_name}"
    )
    
    # DataLoader
    def worker_init_fn(worker_id):
        np.random.seed(SEED + worker_id)
    
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_dataset = ECGDataset(train_data, train_rr, train_labels, train_pid, train_sid)
    valid_dataset = ECGDataset(valid_data, valid_rr, valid_labels, valid_pid, valid_sid)
    test_dataset = ECGDataset(test_data, test_rr, test_labels, test_pid, test_sid)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, generator=g)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    # 모델 생성
    model = get_model(
        exp_name=model_type,
        nOUT=len(CLASSES),
        n_pid=n_records,
        **MODEL_CONFIG
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Best model tracking
    best_auroc = {'value': 0.0, 'epoch': 0, 'state_dict': None}
    best_auprc = {'value': 0.0, 'epoch': 0, 'state_dict': None}
    last_state_dict = None
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics, *_ = train_one_epoch(
            model, train_loader, POLY1_EPS, POLY2_EPS, optimizer, device
        )
        
        if data_config == 'at':
            # Validation 평가
            valid_metrics = evaluate_with_auc(model, valid_loader, device, CLASSES)
            
            # Best AUROC 체크
            if valid_metrics['macro_auroc'] > best_auroc['value']:
                best_auroc = {
                    'value': valid_metrics['macro_auroc'],
                    'epoch': epoch,
                    'state_dict': copy.deepcopy(model.state_dict())
                }
            
            # Best AUPRC 체크
            if valid_metrics['macro_auprc'] > best_auprc['value']:
                best_auprc = {
                    'value': valid_metrics['macro_auprc'],
                    'epoch': epoch,
                    'state_dict': copy.deepcopy(model.state_dict())
                }
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} - AUROC: {valid_metrics['macro_auroc']:.4f}, "
                      f"AUPRC: {valid_metrics['macro_auprc']:.4f}")
        else:
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}")
        
        scheduler.step()
        
        if epoch == EPOCHS:
            last_state_dict = copy.deepcopy(model.state_dict())
    
    # 모델 저장 및 평가
    results = {'exp_name': exp_name, 'model_type': model_type, 'data_config': data_config}
    
    if data_config == 'star':
        # * 실험: 50 epoch 사용 (AUROC/AUPRC 동일)
        best_auroc = {'value': 0.0, 'epoch': EPOCHS, 'state_dict': last_state_dict}
        best_auprc = {'value': 0.0, 'epoch': EPOCHS, 'state_dict': last_state_dict}
    
    # Best AUROC 모델 평가
    model.load_state_dict(best_auroc['state_dict'])
    valid_auroc_metrics = evaluate_with_auc(model, valid_loader, device, CLASSES)
    test_auroc_metrics = evaluate_with_auc(model, test_loader, device, CLASSES)
    
    results['best_auroc'] = {
        'epoch': best_auroc['epoch'],
        'valid': valid_auroc_metrics,
        'test': test_auroc_metrics,
    }
    
    # Best AUPRC 모델 평가
    model.load_state_dict(best_auprc['state_dict'])
    valid_auprc_metrics = evaluate_with_auc(model, valid_loader, device, CLASSES)
    test_auprc_metrics = evaluate_with_auc(model, test_loader, device, CLASSES)
    
    results['best_auprc'] = {
        'epoch': best_auprc['epoch'],
        'valid': valid_auprc_metrics,
        'test': test_auprc_metrics,
    }
    
    # 모델 저장
    torch.save({
        'model_state_dict': best_auroc['state_dict'],
        'epoch': best_auroc['epoch'],
        'auroc': best_auroc['value'],
    }, os.path.join(exp_dir, 'best_weights', f'best_auroc_{exp_name}.pth'))
    
    torch.save({
        'model_state_dict': best_auprc['state_dict'],
        'epoch': best_auprc['epoch'],
        'auprc': best_auprc['value'],
    }, os.path.join(exp_dir, 'best_weights', f'best_auprc_{exp_name}.pth'))
    
    return results, exp_dir


def save_to_template(all_results, template_path, output_path):
    """결과를 템플릿 형식으로 저장"""
    
    # 템플릿 로드
    with pd.ExcelFile(template_path) as xl:
        df_auroc = pd.read_excel(xl, sheet_name='Best AUROC', header=None)
        df_auprc = pd.read_excel(xl, sheet_name='Best AUPRC', header=None)
        df_comp = pd.read_excel(xl, sheet_name='Comparison', header=None)
    
    # 열 매핑 (0-indexed)
    # 0: Experiment, 1: Model, 2: Best Epoch
    # 3: N_AUROC, 4: S_AUROC, 5: V_AUROC, 6: F_AUROC
    # 7: N_AUPRC, 8: S_AUPRC, 9: V_AUPRC, 10: F_AUPRC
    # 11: Accuracy, 12: macro_auroc, 13: macro_auprc, 14: weighted_auroc, 15: weighted_auprc
    
    for exp_name, result in all_results.items():
        if exp_name not in EXP_ROW_MAP:
            continue
        
        row = EXP_ROW_MAP[exp_name]
        
        # Best AUROC 시트
        auroc_valid = result['best_auroc']['valid']
        df_auroc.iloc[row, 2] = result['best_auroc']['epoch']
        df_auroc.iloc[row, 3] = auroc_valid['per_class_auroc'][0]  # N
        df_auroc.iloc[row, 4] = auroc_valid['per_class_auroc'][1]  # S
        df_auroc.iloc[row, 5] = auroc_valid['per_class_auroc'][2]  # V
        df_auroc.iloc[row, 6] = auroc_valid['per_class_auroc'][3]  # F
        df_auroc.iloc[row, 7] = auroc_valid['per_class_auprc'][0]
        df_auroc.iloc[row, 8] = auroc_valid['per_class_auprc'][1]
        df_auroc.iloc[row, 9] = auroc_valid['per_class_auprc'][2]
        df_auroc.iloc[row, 10] = auroc_valid['per_class_auprc'][3]
        df_auroc.iloc[row, 11] = auroc_valid['accuracy']
        df_auroc.iloc[row, 12] = auroc_valid['macro_auroc']
        df_auroc.iloc[row, 13] = auroc_valid['macro_auprc']
        df_auroc.iloc[row, 14] = auroc_valid['weighted_auroc']
        df_auroc.iloc[row, 15] = auroc_valid['weighted_auprc']
        
        # Best AUPRC 시트
        auprc_valid = result['best_auprc']['valid']
        df_auprc.iloc[row, 2] = result['best_auprc']['epoch']
        df_auprc.iloc[row, 3] = auprc_valid['per_class_auroc'][0]
        df_auprc.iloc[row, 4] = auprc_valid['per_class_auroc'][1]
        df_auprc.iloc[row, 5] = auprc_valid['per_class_auroc'][2]
        df_auprc.iloc[row, 6] = auprc_valid['per_class_auroc'][3]
        df_auprc.iloc[row, 7] = auprc_valid['per_class_auprc'][0]
        df_auprc.iloc[row, 8] = auprc_valid['per_class_auprc'][1]
        df_auprc.iloc[row, 9] = auprc_valid['per_class_auprc'][2]
        df_auprc.iloc[row, 10] = auprc_valid['per_class_auprc'][3]
        df_auprc.iloc[row, 11] = auprc_valid['accuracy']
        df_auprc.iloc[row, 12] = auprc_valid['macro_auroc']
        df_auprc.iloc[row, 13] = auprc_valid['macro_auprc']
        df_auprc.iloc[row, 14] = auprc_valid['weighted_auroc']
        df_auprc.iloc[row, 15] = auprc_valid['weighted_auprc']
    
    # Comparison 시트 업데이트
    comparisons = [
        (1, 'A (star)', 'A0*', 'A1*', 'A2*'),
        (2, 'B (star)', 'B0*', 'B1*', 'B2*'),
        (3, 'A (at)', 'A0@', 'A1@', 'A2@'),
        (4, 'B (at)', None, 'B1@', 'B2@'),  # B0@ 없음
    ]
    
    for row, series, base_exp, naive_exp, cross_exp in comparisons:
        base_auroc = all_results.get(base_exp, {}).get('best_auroc', {}).get('test', {}).get('macro_auroc', 0) if base_exp else 0
        naive_auroc = all_results.get(naive_exp, {}).get('best_auroc', {}).get('test', {}).get('macro_auroc', 0) if naive_exp else 0
        cross_auroc = all_results.get(cross_exp, {}).get('best_auroc', {}).get('test', {}).get('macro_auroc', 0) if cross_exp else 0
        
        base_auprc = all_results.get(base_exp, {}).get('best_auroc', {}).get('test', {}).get('macro_auprc', 0) if base_exp else 0
        naive_auprc = all_results.get(naive_exp, {}).get('best_auroc', {}).get('test', {}).get('macro_auprc', 0) if naive_exp else 0
        cross_auprc = all_results.get(cross_exp, {}).get('best_auroc', {}).get('test', {}).get('macro_auprc', 0) if cross_exp else 0
        
        df_comp.iloc[row, 1] = base_auroc
        df_comp.iloc[row, 2] = naive_auroc
        df_comp.iloc[row, 3] = cross_auroc
        df_comp.iloc[row, 4] = naive_auroc - base_auroc
        df_comp.iloc[row, 5] = cross_auroc - naive_auroc
        df_comp.iloc[row, 6] = cross_auroc - base_auroc
        df_comp.iloc[row, 7] = base_auprc
        df_comp.iloc[row, 8] = naive_auprc
        df_comp.iloc[row, 9] = cross_auprc
    
    # 저장
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_auroc.to_excel(writer, sheet_name='Best AUROC', index=False, header=False)
        df_auprc.to_excel(writer, sheet_name='Best AUPRC', index=False, header=False)
        df_comp.to_excel(writer, sheet_name='Comparison', index=False, header=False)
    
    print(f"\n✅ Results saved to: {output_path}")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("Dual Best (AUROC & AUPRC) Experiment")
    print("="*80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print("="*80)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    all_results = {}
    total_start = time.time()
    
    for exp_idx, (exp_name, model_type, data_config) in enumerate(EXPERIMENTS):
        print(f"\n{'='*80}")
        print(f"[{exp_idx+1}/{len(EXPERIMENTS)}] {exp_name}")
        print(f"Model: {model_type}, Config: {data_config}")
        print(f"{'='*80}")
        
        try:
            exp_start = time.time()
            result, exp_dir = run_experiment(exp_name, model_type, data_config, device)
            exp_time = (time.time() - exp_start) / 60
            
            all_results[exp_name] = result
            
            print(f"\n  📊 Results:")
            print(f"     Best AUROC Epoch: {result['best_auroc']['epoch']}")
            print(f"       Valid AUROC: {result['best_auroc']['valid']['macro_auroc']:.4f}")
            print(f"       Test AUROC:  {result['best_auroc']['test']['macro_auroc']:.4f}")
            print(f"     Best AUPRC Epoch: {result['best_auprc']['epoch']}")
            print(f"       Valid AUPRC: {result['best_auprc']['valid']['macro_auprc']:.4f}")
            print(f"       Test AUPRC:  {result['best_auprc']['test']['macro_auprc']:.4f}")
            print(f"     Time: {exp_time:.1f} min")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = (time.time() - total_start) / 60
    
    # 템플릿에 결과 저장
    template_path = 'New_Analysis_template.xlsx'
    if os.path.exists(template_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_xlsx = os.path.join(OUTPUT_PATH, f'New_Analysis_Results_{timestamp}.xlsx')
        save_to_template(all_results, template_path, output_xlsx)
    
    # 최종 요약
    print("\n" + "="*80)
    print("All Experiments Complete!")
    print("="*80)
    print(f"Total time: {total_time:.1f} min ({total_time/60:.1f} hours)")
    
    print(f"\n📊 Summary:")
    print("-"*80)
    print(f"{'Exp':<6} | {'AUROC Ep':>8} | {'Valid AUROC':>11} | {'AUPRC Ep':>8} | {'Valid AUPRC':>11}")
    print("-"*80)
    for exp_name in [e[0] for e in EXPERIMENTS]:
        if exp_name in all_results:
            r = all_results[exp_name]
            print(f"{exp_name:<6} | {r['best_auroc']['epoch']:>8} | "
                  f"{r['best_auroc']['valid']['macro_auroc']:>11.4f} | "
                  f"{r['best_auprc']['epoch']:>8} | "
                  f"{r['best_auprc']['valid']['macro_auprc']:>11.4f}")
    
    print(f"\n✅ All results saved to: {OUTPUT_PATH}")
    print("="*80)

