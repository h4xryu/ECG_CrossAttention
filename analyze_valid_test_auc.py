# analyze_valid_test_auc.py - Valid/Test AUROC, AUPRC 분석 스크립트
# 저장된 모델들의 Validation Set과 Test Set에서의 AUROC, AUPRC 계산
#
# 사용법: python analyze_valid_test_auc.py
#
# 출력 형식:
# Experiment | Model | Data_Config | N_AUROC | S_AUROC | V_AUROC | F_AUROC | 
# N_AUPRC | S_AUPRC | V_AUPRC | F_AUPRC | Accuracy | macro_auroc | macro_auprc | 
# weighted_auroc | weighted_auprc

import os
import glob
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset

# =============================================================================
# 설정
# =============================================================================

DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
AUTO_RESULTS_PATH = './auto_results/'
OUTPUT_PATH = './analysis_results/'
BATCH_SIZE = 1024
SEED = 1234
CLASSES = ['N', 'S', 'V', 'F']

# RR Feature 설정
RR_FEATURE_OPTION = "opt4"
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

# 실험명 -> 모델타입 매핑
EXP_MODEL_MAP = {
    'A0': 'baseline',
    'A1': 'naive_concatenate',
    'A2': 'cross_attention',
    'B0': 'baseline_B',
    'B1': 'naive_concatenate_B',
    'B2': 'cross_attention_B',
}

# =============================================================================
# 평가 함수
# =============================================================================

def evaluate_with_probs(model, data_loader, device):
    """확률값과 함께 평가 수행"""
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
    
    return np.array(y_pred), np.array(y_true), np.array(y_probs)


def calculate_all_metrics(y_true, y_pred, y_probs, classes):
    """모든 메트릭 계산 (Accuracy + AUROC + AUPRC)"""
    n_classes = len(classes)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # One-hot encoding for AUC
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
    
    # Macro average
    macro_auroc = np.mean(per_class_auroc)
    macro_auprc = np.mean(per_class_auprc)
    
    # Weighted average
    class_counts = np.bincount(y_true, minlength=n_classes)
    class_weights = class_counts / len(y_true)
    
    weighted_auroc = np.sum(np.array(per_class_auroc) * class_weights)
    weighted_auprc = np.sum(np.array(per_class_auprc) * class_weights)
    
    return {
        'Accuracy': accuracy,
        'N_AUROC': per_class_auroc[0],
        'S_AUROC': per_class_auroc[1],
        'V_AUROC': per_class_auroc[2],
        'F_AUROC': per_class_auroc[3],
        'N_AUPRC': per_class_auprc[0],
        'S_AUPRC': per_class_auprc[1],
        'V_AUPRC': per_class_auprc[2],
        'F_AUPRC': per_class_auprc[3],
        'macro_auroc': macro_auroc,
        'macro_auprc': macro_auprc,
        'weighted_auroc': weighted_auroc,
        'weighted_auprc': weighted_auprc,
    }


def find_model_path(exp_dir):
    """실험 폴더에서 모델 경로 찾기"""
    best_weights_dir = os.path.join(exp_dir, 'best_weights')
    
    if os.path.exists(best_weights_dir):
        pth_files = glob.glob(os.path.join(best_weights_dir, '*.pth'))
        if pth_files:
            return pth_files[0]
    
    pth_files = glob.glob(os.path.join(exp_dir, '*.pth'))
    if pth_files:
        for pth in pth_files:
            if 'best' in pth.lower() or 'last' in pth.lower():
                return pth
        return pth_files[0]
    
    return None


def get_exp_info(exp_dir_name):
    """실험 폴더명에서 정보 추출"""
    parts = exp_dir_name.split('_')
    if len(parts) >= 1:
        exp_name = parts[0]
        
        if exp_name.endswith('*'):
            exp_base = exp_name[:-1]
            data_config = 'star'
        elif exp_name.endswith('@'):
            exp_base = exp_name[:-1]
            data_config = 'at'
        else:
            exp_base = exp_name[:2] if len(exp_name) >= 2 else exp_name
            data_config = 'unknown'
        
        return exp_name, exp_base, data_config
    
    return None, None, None


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(SEED)
    
    print("="*100)
    print("Valid/Test AUROC & AUPRC Analysis")
    print("="*100)
    print(f"Device: {device}")
    print(f"Results Path: {AUTO_RESULTS_PATH}")
    print("="*100)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # ==========================================================================
    # 데이터 로드
    # ==========================================================================
    print("\n📂 Loading datasets...")
    
    # Validation Set (DS1-2, 5명) - @ 실험의 validation
    valid_data, valid_labels, valid_rr, valid_pid, valid_sid = load_or_extract_data(
        record_list=DS1_VALID_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name="Valid"
    )
    valid_dataset = ECGDataset(valid_data, valid_rr, valid_labels, valid_pid, valid_sid)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"  Valid samples: {len(valid_labels):,} | {dict(Counter(valid_labels))}")
    
    # Test Set (DS2, 22명)
    test_data, test_labels, test_rr, test_pid, test_sid = load_or_extract_data(
        record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name="Test"
    )
    test_dataset = ECGDataset(test_data, test_rr, test_labels, test_pid, test_sid)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"  Test samples: {len(test_labels):,} | {dict(Counter(test_labels))}")
    
    # ==========================================================================
    # 실험 분석
    # ==========================================================================
    exp_dirs = sorted(glob.glob(os.path.join(AUTO_RESULTS_PATH, '*')))
    exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]
    
    if not exp_dirs:
        print(f"\n❌ No experiment directories found in {AUTO_RESULTS_PATH}")
        exit(1)
    
    print(f"\n📁 Found {len(exp_dirs)} experiment directories")
    
    valid_results = []
    test_results = []
    
    for exp_dir in exp_dirs:
        exp_dir_name = os.path.basename(exp_dir)
        exp_name, exp_base, data_config = get_exp_info(exp_dir_name)
        
        if exp_base is None or exp_base not in EXP_MODEL_MAP:
            continue
        
        model_type = EXP_MODEL_MAP[exp_base]
        model_path = find_model_path(exp_dir)
        
        if model_path is None:
            print(f"⚠️  No model found: {exp_dir_name}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📊 {exp_name} | Model: {model_type} | Config: {data_config}")
        print(f"{'='*80}")
        
        try:
            # 모델 로드
            n_records = len(DS1_FULL)
            model = get_model(
                exp_name=model_type,
                nOUT=len(CLASSES),
                n_pid=n_records,
                **MODEL_CONFIG
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            saved_epoch = checkpoint.get('epoch', 'N/A')
            print(f"  Loaded epoch: {saved_epoch}")
            
            # Validation Set 평가
            y_pred_v, y_true_v, y_probs_v = evaluate_with_probs(model, valid_loader, device)
            valid_metrics = calculate_all_metrics(y_true_v, y_pred_v, y_probs_v, CLASSES)
            valid_metrics['Experiment'] = exp_name
            valid_metrics['Model'] = model_type
            valid_metrics['Data_Config'] = data_config
            valid_metrics['Epoch'] = saved_epoch
            valid_results.append(valid_metrics)
            
            # Test Set 평가
            y_pred_t, y_true_t, y_probs_t = evaluate_with_probs(model, test_loader, device)
            test_metrics = calculate_all_metrics(y_true_t, y_pred_t, y_probs_t, CLASSES)
            test_metrics['Experiment'] = exp_name
            test_metrics['Model'] = model_type
            test_metrics['Data_Config'] = data_config
            test_metrics['Epoch'] = saved_epoch
            test_results.append(test_metrics)
            
            # 출력
            print(f"\n  📈 Validation Set:")
            print(f"     Accuracy: {valid_metrics['Accuracy']:.4f}")
            print(f"     Macro AUROC: {valid_metrics['macro_auroc']:.4f} | AUPRC: {valid_metrics['macro_auprc']:.4f}")
            print(f"     Per-class AUROC: N={valid_metrics['N_AUROC']:.4f}, S={valid_metrics['S_AUROC']:.4f}, "
                  f"V={valid_metrics['V_AUROC']:.4f}, F={valid_metrics['F_AUROC']:.4f}")
            
            print(f"\n  📊 Test Set:")
            print(f"     Accuracy: {test_metrics['Accuracy']:.4f}")
            print(f"     Macro AUROC: {test_metrics['macro_auroc']:.4f} | AUPRC: {test_metrics['macro_auprc']:.4f}")
            print(f"     Per-class AUROC: N={test_metrics['N_AUROC']:.4f}, S={test_metrics['S_AUROC']:.4f}, "
                  f"V={test_metrics['V_AUROC']:.4f}, F={test_metrics['F_AUROC']:.4f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ==========================================================================
    # 결과 저장
    # ==========================================================================
    if valid_results and test_results:
        # 컬럼 순서
        col_order = ['Experiment', 'Model', 'Data_Config', 'Epoch',
                     'N_AUROC', 'S_AUROC', 'V_AUROC', 'F_AUROC',
                     'N_AUPRC', 'S_AUPRC', 'V_AUPRC', 'F_AUPRC',
                     'Accuracy', 'macro_auroc', 'macro_auprc', 'weighted_auroc', 'weighted_auprc']
        
        df_valid = pd.DataFrame(valid_results)[col_order]
        df_test = pd.DataFrame(test_results)[col_order]
        
        # 정렬
        df_valid = df_valid.sort_values('Experiment')
        df_test = df_test.sort_values('Experiment')
        
        # 출력
        print("\n" + "="*120)
        print("📊 Validation Set Results")
        print("="*120)
        print(df_valid.to_string(index=False))
        
        print("\n" + "="*120)
        print("📊 Test Set Results")
        print("="*120)
        print(df_test.to_string(index=False))
        
        # Excel 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(OUTPUT_PATH, f'Valid_Test_AUC_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_valid.to_excel(writer, sheet_name='Validation', index=False)
            df_test.to_excel(writer, sheet_name='Test', index=False)
            
            # Comparison sheet
            comparison_data = []
            for config in ['star', 'at']:
                for series in ['A', 'B']:
                    baseline_exp = f"{series}0{'*' if config == 'star' else '@'}"
                    naive_exp = f"{series}1{'*' if config == 'star' else '@'}"
                    cross_exp = f"{series}2{'*' if config == 'star' else '@'}"
                    
                    baseline_t = next((r for r in test_results if r['Experiment'] == baseline_exp), None)
                    naive_t = next((r for r in test_results if r['Experiment'] == naive_exp), None)
                    cross_t = next((r for r in test_results if r['Experiment'] == cross_exp), None)
                    
                    if baseline_t and naive_t and cross_t:
                        comparison_data.append({
                            'Series': f"{series} ({config})",
                            'Baseline_AUROC': baseline_t['macro_auroc'],
                            'Naive_AUROC': naive_t['macro_auroc'],
                            'Cross_AUROC': cross_t['macro_auroc'],
                            'Naive_vs_Base': naive_t['macro_auroc'] - baseline_t['macro_auroc'],
                            'Cross_vs_Naive': cross_t['macro_auroc'] - naive_t['macro_auroc'],
                            'Cross_vs_Base': cross_t['macro_auroc'] - baseline_t['macro_auroc'],
                            'Baseline_AUPRC': baseline_t['macro_auprc'],
                            'Naive_AUPRC': naive_t['macro_auprc'],
                            'Cross_AUPRC': cross_t['macro_auprc'],
                        })
            
            if comparison_data:
                df_comp = pd.DataFrame(comparison_data)
                df_comp.to_excel(writer, sheet_name='Comparison', index=False)
        
        print(f"\n✅ Results saved to: {excel_path}")
        
        # 비교 출력
        if comparison_data:
            print("\n" + "="*100)
            print("📊 Model Comparison (Test Set - Macro AUROC)")
            print("="*100)
            print(f"{'Series':<15} {'Baseline':>10} {'Naive':>10} {'Cross':>10} {'N-B':>10} {'C-N':>10} {'C-B':>10}")
            print("-"*100)
            for c in comparison_data:
                print(f"{c['Series']:<15} {c['Baseline_AUROC']:>10.4f} {c['Naive_AUROC']:>10.4f} "
                      f"{c['Cross_AUROC']:>10.4f} {c['Naive_vs_Base']:>+10.4f} "
                      f"{c['Cross_vs_Naive']:>+10.4f} {c['Cross_vs_Base']:>+10.4f}")
    
    print("\n" + "="*100)
    print("Analysis Complete!")
    print("="*100)

