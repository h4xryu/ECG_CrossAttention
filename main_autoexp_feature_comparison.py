# main_autoexp_feature_comparison.py - RR Feature Set 비교 실험 스크립트
# opt1, opt2, opt3, opt4 다양한 RR feature set으로 학습하고 비교
#
# 사용법: python main_autoexp_feature_comparison.py
#
# 출력:
#   - 각 feature set별 모든 메트릭 (Accuracy, AUROC, AUPRC)
#   - Per-class AUROC, AUPRC
#   - 비교 테이블

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset
from train import train_one_epoch, validate

# =============================================================================
# 설정
# =============================================================================

DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH = './feature_comparison_results/'
BATCH_SIZE = 1024
EPOCHS = 50
LR = 0.0001
WEIGHT_DECAY = 1e-3
SEED = 1234
POLY1_EPS = 0.0
POLY2_EPS = 0.0
CLASSES = ['N', 'S', 'V', 'F']

# RR Feature 설정
RR_FEATURE_OPTIONS = ['opt1', 'opt2', 'opt3', 'opt4']
RR_FEATURE_DIMS = {"opt1": 7, "opt2": 38, "opt3": 7, "opt4": 7}

# ECG Parameters
VALID_LEADS = ['MLII', 'V1', 'V2', 'V4', 'V5']
OUT_LEN = 720

# =============================================================================
# 데이터 분할 설정
# =============================================================================

DS1_FULL = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119',
    '122', '201', '203', '209', '215', '223', '230', '208',
    '114', '124', '205', '207', '220'
]

DS1_TRAIN_SPLIT = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119',
    '122', '201', '203', '209', '215', '223', '230', '208'
]
DS1_VALID_SPLIT = ['114', '124', '205', '207', '220']

DS2_TEST = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]

# 비교할 실험 (Cross-Attention 모델로 feature set 비교)
EXPERIMENTS = [
    ('A2@', 'cross_attention', 'at'),  # Dense + Cross-Attention
    ('B2@', 'cross_attention_B', 'at'),  # No Dense + Cross-Attention
]

# =============================================================================
# 평가 함수
# =============================================================================

def evaluate_with_full_metrics(model, data_loader, device, classes):
    """모든 메트릭과 함께 평가"""
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
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # One-hot for AUC
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1
    
    # Per-class AUROC, AUPRC
    per_class_auroc = []
    per_class_auprc = []
    
    for i in range(n_classes):
        try:
            auroc = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
        except ValueError:
            auroc = 0.0
        try:
            auprc = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
        except ValueError:
            auprc = 0.0
        per_class_auroc.append(auroc)
        per_class_auprc.append(auprc)
    
    # Macro/Weighted averages
    class_counts = np.bincount(y_true, minlength=n_classes)
    class_weights = class_counts / len(y_true)
    
    macro_auroc = np.mean(per_class_auroc)
    macro_auprc = np.mean(per_class_auprc)
    weighted_auroc = np.sum(np.array(per_class_auroc) * class_weights)
    weighted_auprc = np.sum(np.array(per_class_auprc) * class_weights)
    
    macro_prec = np.mean(per_class_precision)
    macro_recall = np.mean(per_class_recall)
    macro_f1 = np.mean(per_class_f1)
    
    return {
        'accuracy': accuracy,
        'macro_prec': macro_prec,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_auroc': macro_auroc,
        'macro_auprc': macro_auprc,
        'weighted_auroc': weighted_auroc,
        'weighted_auprc': weighted_auprc,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_auroc': per_class_auroc,
        'per_class_auprc': per_class_auprc,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }


def run_feature_experiment(exp_name, model_type, data_config, rr_option, device):
    """단일 feature set으로 실험 수행"""
    
    set_seed(SEED)
    
    n_rr = RR_FEATURE_DIMS[rr_option]
    
    # 모델 설정
    model_config = {
        'in_channels': 1,
        'out_ch': 180,
        'mid_ch': 30,
        'num_heads': 9,
        'n_rr': n_rr,
    }
    
    # 데이터 로드 (해당 RR feature option으로)
    # Note: load_or_extract_data에서 RR feature option을 설정해야 함
    # 여기서는 config.py의 설정을 따르므로, 런타임에 변경이 필요
    
    from config import RR_FEATURE_OPTION as CONFIG_RR_OPTION
    import config
    original_option = config.RR_FEATURE_OPTION
    config.RR_FEATURE_OPTION = rr_option
    
    if data_config == 'star':
        train_data, train_labels, train_rr, train_pid, train_sid = load_or_extract_data(
            record_list=DS1_FULL, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Train_{rr_option}"
        )
        valid_data, valid_labels, valid_rr, valid_pid, valid_sid = \
            train_data, train_labels, train_rr, train_pid, train_sid
        n_records = len(DS1_FULL)
    else:  # 'at'
        train_data, train_labels, train_rr, train_pid, train_sid = load_or_extract_data(
            record_list=DS1_TRAIN_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Train_{rr_option}"
        )
        valid_data, valid_labels, valid_rr, valid_pid, valid_sid = load_or_extract_data(
            record_list=DS1_VALID_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Valid_{rr_option}"
        )
        n_records = len(DS1_TRAIN_SPLIT) + len(DS1_VALID_SPLIT)
    
    test_data, test_labels, test_rr, test_pid, test_sid = load_or_extract_data(
        record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name=f"Test_{rr_option}"
    )
    
    config.RR_FEATURE_OPTION = original_option
    
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
        **model_config
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Best model tracking
    best_auroc = {'value': 0.0, 'epoch': 0, 'state_dict': None}
    
    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics, *_ = train_one_epoch(
            model, train_loader, POLY1_EPS, POLY2_EPS, optimizer, device
        )
        
        if data_config == 'at':
            valid_loss, valid_metrics, *_ = validate(
                model, valid_loader, POLY1_EPS, POLY2_EPS, device
            )
            
            if valid_metrics['macro_auroc'] > best_auroc['value']:
                best_auroc = {
                    'value': valid_metrics['macro_auroc'],
                    'epoch': epoch,
                    'state_dict': copy.deepcopy(model.state_dict())
                }
        
        scheduler.step()
        
        if epoch == EPOCHS and data_config == 'star':
            best_auroc = {
                'value': 0.0,
                'epoch': epoch,
                'state_dict': copy.deepcopy(model.state_dict())
            }
    
    # Load best model and evaluate
    model.load_state_dict(best_auroc['state_dict'])
    
    valid_metrics = evaluate_with_full_metrics(model, valid_loader, device, CLASSES)
    test_metrics = evaluate_with_full_metrics(model, test_loader, device, CLASSES)
    
    return {
        'best_epoch': best_auroc['epoch'],
        'valid': valid_metrics,
        'test': test_metrics,
    }


# =============================================================================
# 결과 저장
# =============================================================================

def save_comparison_results(all_results, output_path):
    """비교 결과를 엑셀로 저장"""
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Summary sheet
        summary_data = []
        for (exp_name, rr_option), result in all_results.items():
            test = result['test']
            summary_data.append({
                'Experiment': exp_name,
                'RR_Feature': rr_option,
                'RR_Dim': RR_FEATURE_DIMS[rr_option],
                'Best_Epoch': result['best_epoch'],
                'Accuracy': test['accuracy'],
                'Macro_F1': test['macro_f1'],
                'Macro_AUROC': test['macro_auroc'],
                'Macro_AUPRC': test['macro_auprc'],
                'Weighted_AUROC': test['weighted_auroc'],
                'Weighted_AUPRC': test['weighted_auprc'],
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values(['Experiment', 'RR_Feature'])
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Per-class sheet
        per_class_data = []
        for (exp_name, rr_option), result in all_results.items():
            test = result['test']
            for i, cls in enumerate(CLASSES):
                per_class_data.append({
                    'Experiment': exp_name,
                    'RR_Feature': rr_option,
                    'Class': cls,
                    'Precision': test['per_class_precision'][i],
                    'Recall': test['per_class_recall'][i],
                    'F1': test['per_class_f1'][i],
                    'AUROC': test['per_class_auroc'][i],
                    'AUPRC': test['per_class_auprc'][i],
                })
        
        df_per_class = pd.DataFrame(per_class_data)
        df_per_class.to_excel(writer, sheet_name='Per-Class', index=False)
        
        # Validation sheet
        valid_data = []
        for (exp_name, rr_option), result in all_results.items():
            valid = result['valid']
            valid_data.append({
                'Experiment': exp_name,
                'RR_Feature': rr_option,
                'Accuracy': valid['accuracy'],
                'Macro_F1': valid['macro_f1'],
                'Macro_AUROC': valid['macro_auroc'],
                'Macro_AUPRC': valid['macro_auprc'],
                'N_AUROC': valid['per_class_auroc'][0],
                'S_AUROC': valid['per_class_auroc'][1],
                'V_AUROC': valid['per_class_auroc'][2],
                'F_AUROC': valid['per_class_auroc'][3],
            })
        
        df_valid = pd.DataFrame(valid_data)
        df_valid.to_excel(writer, sheet_name='Validation', index=False)
        
        # Feature comparison sheet
        comparison_data = []
        for exp_name in [e[0] for e in EXPERIMENTS]:
            exp_results = {rr: all_results.get((exp_name, rr)) for rr in RR_FEATURE_OPTIONS}
            
            if all(v is not None for v in exp_results.values()):
                comparison_data.append({
                    'Experiment': exp_name,
                    'opt1_AUROC': exp_results['opt1']['test']['macro_auroc'],
                    'opt2_AUROC': exp_results['opt2']['test']['macro_auroc'],
                    'opt3_AUROC': exp_results['opt3']['test']['macro_auroc'],
                    'opt4_AUROC': exp_results['opt4']['test']['macro_auroc'],
                    'opt1_AUPRC': exp_results['opt1']['test']['macro_auprc'],
                    'opt2_AUPRC': exp_results['opt2']['test']['macro_auprc'],
                    'opt3_AUPRC': exp_results['opt3']['test']['macro_auprc'],
                    'opt4_AUPRC': exp_results['opt4']['test']['macro_auprc'],
                    'Best_Feature': max(RR_FEATURE_OPTIONS, 
                                       key=lambda x: exp_results[x]['test']['macro_auroc']),
                })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            df_comp.to_excel(writer, sheet_name='Feature_Comparison', index=False)
    
    print(f"\n✅ Results saved to: {output_path}")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("RR Feature Set Comparison Experiment")
    print("="*80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Feature Options: {RR_FEATURE_OPTIONS}")
    print(f"Experiments: {[e[0] for e in EXPERIMENTS]}")
    print(f"Total runs: {len(EXPERIMENTS) * len(RR_FEATURE_OPTIONS)}")
    print("="*80)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    all_results = {}
    total_start = time.time()
    
    for exp_idx, (exp_name, model_type, data_config) in enumerate(EXPERIMENTS):
        for rr_idx, rr_option in enumerate(RR_FEATURE_OPTIONS):
            run_idx = exp_idx * len(RR_FEATURE_OPTIONS) + rr_idx + 1
            total_runs = len(EXPERIMENTS) * len(RR_FEATURE_OPTIONS)
            
            print(f"\n{'='*80}")
            print(f"[{run_idx}/{total_runs}] {exp_name} with {rr_option} (dim={RR_FEATURE_DIMS[rr_option]})")
            print(f"{'='*80}")
            
            try:
                exp_start = time.time()
                result = run_feature_experiment(exp_name, model_type, data_config, rr_option, device)
                exp_time = (time.time() - exp_start) / 60
                
                all_results[(exp_name, rr_option)] = result
                
                print(f"\n  📊 Results (Best Epoch: {result['best_epoch']}):")
                print(f"     Test Accuracy:   {result['test']['accuracy']:.4f}")
                print(f"     Test Macro F1:   {result['test']['macro_f1']:.4f}")
                print(f"     Test Macro AUROC: {result['test']['macro_auroc']:.4f}")
                print(f"     Test Macro AUPRC: {result['test']['macro_auprc']:.4f}")
                print(f"\n     Per-class AUROC: N={result['test']['per_class_auroc'][0]:.4f}, "
                      f"S={result['test']['per_class_auroc'][1]:.4f}, "
                      f"V={result['test']['per_class_auroc'][2]:.4f}, "
                      f"F={result['test']['per_class_auroc'][3]:.4f}")
                print(f"\n     Time: {exp_time:.1f} min")
                
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    total_time = (time.time() - total_start) / 60
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_xlsx = os.path.join(OUTPUT_PATH, f'Feature_Comparison_{timestamp}.xlsx')
    save_comparison_results(all_results, output_xlsx)
    
    # 최종 요약
    print("\n" + "="*80)
    print("All Experiments Complete!")
    print("="*80)
    print(f"Total time: {total_time:.1f} min ({total_time/60:.1f} hours)")
    
    print(f"\n📊 Summary (Test Macro AUROC):")
    print("-"*60)
    print(f"{'Experiment':<10} | {'opt1':>8} | {'opt2':>8} | {'opt3':>8} | {'opt4':>8}")
    print("-"*60)
    
    for exp_name in [e[0] for e in EXPERIMENTS]:
        row = f"{exp_name:<10}"
        for rr_option in RR_FEATURE_OPTIONS:
            key = (exp_name, rr_option)
            if key in all_results:
                auroc = all_results[key]['test']['macro_auroc']
                row += f" | {auroc:>8.4f}"
            else:
                row += f" | {'N/A':>8}"
        print(row)
    
    print(f"\n✅ All results saved to: {output_xlsx}")
    print("="*80)

