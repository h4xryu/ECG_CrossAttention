# analyze_autoexp_results.py - 자동실험 결과 분석 스크립트
# main_autoexp에서 학습된 모델들의 AUROC, AUPRC 등 상세 지표 계산
#
# 사용법: python analyze_autoexp_results.py
# 
# 출력:
#   - 각 실험별 Macro/Weighted AUROC, AUPRC
#   - Per-class AUROC, AUPRC
#   - 비교 테이블

import os
import glob
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset

# =============================================================================
# 설정
# =============================================================================

DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
AUTO_RESULTS_PATH = './auto_results/'  # main_autoexp 결과 경로
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

# 데이터 분할
DS1_FULL = [
    '101', '106', '108', '109', '112', '115', '116', '118', '119',
    '122', '201', '203', '209', '215', '223', '230', '208',
    '114', '124', '205', '207', '220'
]
DS2_TEST = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234'
]

# 실험 정의 (실험명 -> 모델타입 매핑)
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

def evaluate_with_probs(model, test_loader, device):
    """
    확률값과 함께 평가 수행
    
    Returns:
        y_pred: 예측 레이블
        y_true: 실제 레이블
        y_probs: 클래스별 확률 (N x num_classes)
    """
    model.eval()
    y_pred, y_true, y_probs = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
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


def calculate_auc_metrics(y_true, y_probs, classes):
    """
    AUROC, AUPRC 계산
    
    Args:
        y_true: 실제 레이블 (N,)
        y_probs: 클래스별 확률 (N, num_classes)
        classes: 클래스 이름 리스트
    
    Returns:
        metrics dict
    """
    n_classes = len(classes)
    
    # One-hot encoding
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
    
    # Multi-class AUROC (OvR)
    try:
        ovr_auroc = roc_auc_score(y_true_onehot, y_probs, average='macro', multi_class='ovr')
    except ValueError:
        ovr_auroc = macro_auroc
    
    return {
        'macro_auroc': macro_auroc,
        'macro_auprc': macro_auprc,
        'weighted_auroc': weighted_auroc,
        'weighted_auprc': weighted_auprc,
        'ovr_auroc': ovr_auroc,
        'per_class_auroc': per_class_auroc,
        'per_class_auprc': per_class_auprc,
    }


def find_model_path(exp_dir):
    """실험 폴더에서 best model 경로 찾기"""
    best_weights_dir = os.path.join(exp_dir, 'best_weights')
    
    if os.path.exists(best_weights_dir):
        # best_weights 폴더에서 찾기
        pth_files = glob.glob(os.path.join(best_weights_dir, '*.pth'))
        if pth_files:
            return pth_files[0]
    
    # 최상위 폴더에서 찾기
    pth_files = glob.glob(os.path.join(exp_dir, '*.pth'))
    if pth_files:
        # best model 우선
        for pth in pth_files:
            if 'best' in pth.lower() or 'last' in pth.lower():
                return pth
        return pth_files[0]
    
    return None


def get_exp_info(exp_dir_name):
    """
    실험 폴더명에서 실험 정보 추출
    예: A0*_20260120_123456 -> ('A0*', 'A0', 'star')
    """
    parts = exp_dir_name.split('_')
    if len(parts) >= 1:
        exp_name = parts[0]  # A0*, A1@, etc.
        
        # 실험 타입 추출
        if exp_name.endswith('*'):
            exp_base = exp_name[:-1]  # A0, A1, etc.
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
    
    print("="*80)
    print("Auto Experiment Results Analysis (AUROC, AUPRC)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Results Path: {AUTO_RESULTS_PATH}")
    print("="*80)
    
    # 출력 폴더 생성
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 테스트 데이터 로드
    print("\n📂 Loading test data...")
    test_data, test_labels, test_rr, test_pid, test_sid = load_or_extract_data(
        record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name="Test"
    )
    
    test_dataset = ECGDataset(test_data, test_rr, test_labels, test_pid, test_sid)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    print(f"  Test samples: {len(test_labels):,}")
    print(f"  Distribution: {dict(Counter(test_labels))}")
    
    # 실험 폴더 찾기
    exp_dirs = sorted(glob.glob(os.path.join(AUTO_RESULTS_PATH, '*')))
    exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]
    
    if not exp_dirs:
        print(f"\n❌ No experiment directories found in {AUTO_RESULTS_PATH}")
        exit(1)
    
    print(f"\n📁 Found {len(exp_dirs)} experiment directories")
    
    # 각 실험 분석
    all_results = []
    
    for exp_dir in exp_dirs:
        exp_dir_name = os.path.basename(exp_dir)
        exp_name, exp_base, data_config = get_exp_info(exp_dir_name)
        
        if exp_base is None or exp_base not in EXP_MODEL_MAP:
            print(f"\n⚠️  Skipping unknown experiment: {exp_dir_name}")
            continue
        
        model_type = EXP_MODEL_MAP[exp_base]
        model_path = find_model_path(exp_dir)
        
        if model_path is None:
            print(f"\n⚠️  No model found in: {exp_dir_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"📊 Analyzing: {exp_name}")
        print(f"   Model: {model_type}")
        print(f"   Path: {model_path}")
        print(f"{'='*60}")
        
        try:
            # 모델 로드
            n_records = len(DS1_FULL)  # 22명
            model = get_model(
                exp_name=model_type,
                nOUT=len(CLASSES),
                n_pid=n_records,
                **MODEL_CONFIG
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 평가
            y_pred, y_true, y_probs = evaluate_with_probs(model, test_loader, device)
            
            # AUC 메트릭 계산
            auc_metrics = calculate_auc_metrics(y_true, y_probs, CLASSES)
            
            # 정확도 계산
            accuracy = np.mean(y_pred == y_true)
            
            # 결과 저장
            result = {
                'Experiment': exp_name,
                'Model': model_type,
                'Data_Config': data_config,
                'Accuracy': accuracy,
                **auc_metrics
            }
            all_results.append(result)
            
            # 출력
            print(f"\n  Accuracy:      {accuracy:.4f}")
            print(f"  Macro AUROC:   {auc_metrics['macro_auroc']:.4f}")
            print(f"  Macro AUPRC:   {auc_metrics['macro_auprc']:.4f}")
            print(f"  Weighted AUROC: {auc_metrics['weighted_auroc']:.4f}")
            print(f"  Weighted AUPRC: {auc_metrics['weighted_auprc']:.4f}")
            print(f"\n  Per-class AUROC:")
            for i, cls in enumerate(CLASSES):
                print(f"    {cls}: AUROC={auc_metrics['per_class_auroc'][i]:.4f}, "
                      f"AUPRC={auc_metrics['per_class_auprc'][i]:.4f}")
            
        except Exception as e:
            print(f"\n❌ Error analyzing {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 결과 요약
    if all_results:
        print("\n" + "="*80)
        print("📊 Summary Table")
        print("="*80)
        
        # DataFrame 생성
        df = pd.DataFrame(all_results)
        
        # 주요 컬럼만 선택
        summary_cols = ['Experiment', 'Model', 'Data_Config', 'Accuracy', 
                        'macro_auroc', 'macro_auprc', 'weighted_auroc', 'weighted_auprc']
        df_summary = df[summary_cols].copy()
        
        # 소수점 정리
        for col in ['Accuracy', 'macro_auroc', 'macro_auprc', 'weighted_auroc', 'weighted_auprc']:
            df_summary[col] = df_summary[col].apply(lambda x: f"{x:.4f}")
        
        print(df_summary.to_string(index=False))
        
        # Excel 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(OUTPUT_PATH, f'AUC_Analysis_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Per-class sheet
            per_class_data = []
            for result in all_results:
                for i, cls in enumerate(CLASSES):
                    per_class_data.append({
                        'Experiment': result['Experiment'],
                        'Class': cls,
                        'AUROC': result['per_class_auroc'][i],
                        'AUPRC': result['per_class_auprc'][i],
                    })
            df_per_class = pd.DataFrame(per_class_data)
            df_per_class.to_excel(writer, sheet_name='Per-Class', index=False)
            
            # Comparison sheet (Cross-Attention vs Naive Concat)
            comparison_data = []
            for config in ['star', 'at']:
                for series in ['A', 'B']:
                    naive_exp = f"{series}1{'*' if config == 'star' else '@'}"
                    cross_exp = f"{series}2{'*' if config == 'star' else '@'}"
                    
                    naive_result = next((r for r in all_results if r['Experiment'] == naive_exp), None)
                    cross_result = next((r for r in all_results if r['Experiment'] == cross_exp), None)
                    
                    if naive_result and cross_result:
                        comparison_data.append({
                            'Comparison': f"{naive_exp} vs {cross_exp}",
                            'Naive_AUROC': naive_result['macro_auroc'],
                            'Cross_AUROC': cross_result['macro_auroc'],
                            'AUROC_Diff': cross_result['macro_auroc'] - naive_result['macro_auroc'],
                            'Naive_AUPRC': naive_result['macro_auprc'],
                            'Cross_AUPRC': cross_result['macro_auprc'],
                            'AUPRC_Diff': cross_result['macro_auprc'] - naive_result['macro_auprc'],
                            'Naive_Acc': naive_result['Accuracy'],
                            'Cross_Acc': cross_result['Accuracy'],
                            'Acc_Diff': cross_result['Accuracy'] - naive_result['Accuracy'],
                        })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                df_comparison.to_excel(writer, sheet_name='Comparison', index=False)
        
        print(f"\n✅ Results saved to: {excel_path}")
        
        # 비교 결과 출력
        if comparison_data:
            print("\n" + "="*80)
            print("📊 Naive Concat vs Cross-Attention Comparison")
            print("="*80)
            for comp in comparison_data:
                diff_str = f"+{comp['AUROC_Diff']:.4f}" if comp['AUROC_Diff'] > 0 else f"{comp['AUROC_Diff']:.4f}"
                better = "Cross ✓" if comp['AUROC_Diff'] > 0 else "Naive ✗"
                print(f"  {comp['Comparison']:15s}: AUROC diff = {diff_str} ({better})")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)

