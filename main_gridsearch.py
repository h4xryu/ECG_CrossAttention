"""
main_gridsearch.py - Improved Model Hyperparameter Grid Search
새로운 Improved 모델의 최적 하이퍼파라미터를 찾기 위한 그리드 서치

파이프라인:
1. Baseline 모델로 out_ch, num_heads, mid_ch 그리드 서치
2. Validation AUROC 기준으로 최적 조합 선택
3. 최적 파라미터로 B0, B1, B2 실험 실행
"""

import os
import time
import copy
import json
from collections import Counter
from datetime import datetime
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import set_seed, load_or_extract_data
from model import get_model
from dataloader import ECGDataset
from train import train_one_epoch, validate
from evaluate_module import evaluate, calculate_metrics


# =============================================================================
# 설정
# =============================================================================

DATA_PATH = './data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH = './gridsearch_results/'
BATCH_SIZE = 1024
EPOCHS = 30
LR = 0.0001
WEIGHT_DECAY = 1e-3
SEED = 1234
POLY1_EPS = 0.0
POLY2_EPS = 0.0
CLASSES = ['N', 'S', 'V', 'F']

# RR Feature 설정
RR_FEATURE_OPTION = "opt3"
RR_FEATURE_DIMS = {"opt1": 7, "opt2": 38, "opt3": 7, "opt4": 7}

# ECG Parameters
VALID_LEADS = ['MLII', 'V1', 'V2', 'V4', 'V5']
OUT_LEN = 720

# 데이터 분할
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

# =============================================================================
# 그리드 서치 파라미터
# =============================================================================

# 테스트할 하이퍼파라미터 조합
GRID_PARAMS = {
    'out_ch': [32, 48, 64],           # Multi-kernel block output channels
    'num_heads': [3, 4, 6],           # Attention heads
    'mid_ch': [16, 24],               # Mid channels (사용 안 되지만 호환성 유지)
}

# 그리드 생성: 모든 조합
GRID_COMBINATIONS = list(product(
    GRID_PARAMS['out_ch'],
    GRID_PARAMS['num_heads'],
    GRID_PARAMS['mid_ch']
))


# =============================================================================
# 핵심 함수
# =============================================================================

def create_gridsearch_dir():
    """그리드 서치용 폴더 생성"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = os.path.join(OUTPUT_PATH, f'gridsearch_{timestamp}')
    os.makedirs(search_dir, exist_ok=True)
    return search_dir


def run_gridsearch_trial(out_ch, num_heads, mid_ch, trial_num, total_trials, device):
    """
    단일 그리드 서치 시행
    
    Args:
        out_ch: Output channels for multi-kernel blocks
        num_heads: Number of attention heads
        mid_ch: Mid channels
        trial_num: Current trial number
        total_trials: Total trials
        device: torch device
    
    Returns:
        result_dict: {
            'out_ch': out_ch,
            'num_heads': num_heads,
            'mid_ch': mid_ch,
            'trial_num': trial_num,
            'best_valid_auroc': float,
            'best_epoch': int,
            'test_auroc': float,
            'test_auprc': float,
            'test_macro_f1': float,
            'test_accuracy': float,
            'status': 'success' or 'error'
        }
    """
    print(f"\n{'='*80}")
    print(f"Trial {trial_num}/{total_trials}")
    print(f"  Params: out_ch={out_ch}, num_heads={num_heads}, mid_ch={mid_ch}")
    print(f"{'='*80}")
    
    result_dict = {
        'out_ch': out_ch,
        'num_heads': num_heads,
        'mid_ch': mid_ch,
        'trial_num': trial_num,
        'best_valid_auroc': 0.0,
        'best_epoch': 0,
        'test_auroc': 0.0,
        'test_auprc': 0.0,
        'test_macro_f1': 0.0,
        'test_accuracy': 0.0,
        'status': 'error'
    }
    
    try:
        set_seed(SEED)
        
        # 데이터 로드
        print("  Loading data...")
        train_data, train_labels, train_rr, train_pid, train_sid = load_or_extract_data(
            record_list=DS1_TRAIN_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Train_gridsearch"
        )
        valid_data, valid_labels, valid_rr, valid_pid, valid_sid = load_or_extract_data(
            record_list=DS1_VALID_SPLIT, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Valid_gridsearch"
        )
        test_data, test_labels, test_rr, test_pid, test_sid = load_or_extract_data(
            record_list=DS2_TEST, base_path=DATA_PATH, valid_leads=VALID_LEADS,
            out_len=OUT_LEN, split_name=f"Test_gridsearch"
        )
        
        n_records = len(DS1_TRAIN_SPLIT) + len(DS1_VALID_SPLIT)
        
        # DataLoader 생성
        train_dataset = ECGDataset(train_data, train_rr, train_labels, train_pid, train_sid)
        valid_dataset = ECGDataset(valid_data, valid_rr, valid_labels, valid_pid, valid_sid)
        test_dataset = ECGDataset(test_data, test_rr, test_labels, test_pid, test_sid)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=4, pin_memory=True)

        print(f"  Train: {len(train_labels):,} | Valid: {len(valid_labels):,} | Test: {len(test_labels):,}")
        
        # 모델 생성
        model_config = {
            'in_channels': 1,
            'out_ch': out_ch,
            'mid_ch': mid_ch,
            'num_heads': num_heads,
            'n_rr': RR_FEATURE_DIMS[RR_FEATURE_OPTION],
        }
        
        model = get_model(
            exp_name='baseline',
            nOUT=len(CLASSES),
            n_pid=n_records,
            **model_config
        ).to(device)
        
        # 학습 설정
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Best validation AUROC 추적
        best_valid_auroc = {'value': 0.0, 'epoch': 0, 'state_dict': None}
        
        # 학습 루프
        for epoch in range(1, EPOCHS + 1):
            # Train
            train_loss, train_metrics, *_ = train_one_epoch(
                model, train_loader, POLY1_EPS, POLY2_EPS, optimizer, device
            )
            
            # Validation
            valid_loss, valid_metrics, *_ = validate(
                model, valid_loader, POLY1_EPS, POLY2_EPS, device
            )
            
            # Best AUROC 체크
            current_auroc = valid_metrics['macro_auroc']
            if current_auroc > best_valid_auroc['value']:
                best_valid_auroc = {
                    'value': current_auroc,
                    'epoch': epoch,
                    'state_dict': copy.deepcopy(model.state_dict())
                }
            
            if epoch % 5 == 0:
                print(f"    Epoch {epoch}/{EPOCHS} - Valid AUROC: {current_auroc:.4f} "
                      f"(Best: {best_valid_auroc['value']:.4f}@E{best_valid_auroc['epoch']})")
            
            scheduler.step()
        
        # Best 모델로 테스트
        print(f"  Testing with best model (epoch {best_valid_auroc['epoch']})...")
        model.load_state_dict(best_valid_auroc['state_dict'])
        model.eval()
        
        y_pred, y_true, _ = evaluate(model, test_loader, device)
        test_metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
        
        # 결과 저장
        result_dict['best_valid_auroc'] = best_valid_auroc['value']
        result_dict['best_epoch'] = best_valid_auroc['epoch']
        result_dict['test_auroc'] = test_metrics.get('macro_auroc', 0.0)
        result_dict['test_auprc'] = test_metrics.get('macro_auprc', 0.0)
        result_dict['test_macro_f1'] = test_metrics.get('macro_f1', 0.0)
        result_dict['test_accuracy'] = test_metrics.get('overall_accuracy', 0.0)
        result_dict['status'] = 'success'
        
        print(f"  ✓ Test AUROC: {result_dict['test_auroc']:.4f}, "
              f"F1: {result_dict['test_macro_f1']:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        result_dict['status'] = 'error'
    
    return result_dict


def run_gridsearch(device):
    """
    전체 그리드 서치 실행
    
    Args:
        device: torch device
    
    Returns:
        results_list: 모든 시행 결과 리스트
        best_result: 가장 좋은 결과 (best_valid_auroc 기준)
        search_dir: 그리드 서치 결과 디렉토리
    """
    search_dir = create_gridsearch_dir()
    print(f"\n{'='*80}")
    print(f"GRID SEARCH: Improved Deep Residual CNN - Baseline Model")
    print(f"{'='*80}")
    print(f"Output: {search_dir}")
    print(f"Grid combinations: {len(GRID_COMBINATIONS)}")
    print(f"  out_ch: {GRID_PARAMS['out_ch']}")
    print(f"  num_heads: {GRID_PARAMS['num_heads']}")
    print(f"  mid_ch: {GRID_PARAMS['mid_ch']}")
    print(f"{'='*80}")
    
    results_list = []
    total_start = time.time()
    
    # 모든 조합 시행
    for trial_num, (out_ch, num_heads, mid_ch) in enumerate(GRID_COMBINATIONS, 1):
        result_dict = run_gridsearch_trial(
            out_ch=out_ch,
            num_heads=num_heads,
            mid_ch=mid_ch,
            trial_num=trial_num,
            total_trials=len(GRID_COMBINATIONS),
            device=device
        )
        results_list.append(result_dict)
    
    # 최고 성능 찾기 (valid AUROC 기준)
    successful_results = [r for r in results_list if r['status'] == 'success']
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['best_valid_auroc'])
    else:
        best_result = None
    
    # 결과 저장
    total_time = (time.time() - total_start) / 60
    
    summary = {
        'total_trials': len(GRID_COMBINATIONS),
        'successful_trials': len(successful_results),
        'total_time_minutes': total_time,
        'best_result': best_result,
        'all_results': results_list
    }
    
    # JSON 저장
    summary_file = os.path.join(search_dir, 'gridsearch_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # CSV 저장 (가독성)
    import csv
    csv_file = os.path.join(search_dir, 'gridsearch_results.csv')
    if successful_results:
        fieldnames = ['trial_num', 'out_ch', 'num_heads', 'mid_ch', 
                     'best_valid_auroc', 'best_epoch', 'test_auroc', 
                     'test_auprc', 'test_macro_f1', 'test_accuracy', 'status']
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_list)
    
    # 최종 요약 출력
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f} min")
    print(f"Successful trials: {len(successful_results)}/{len(GRID_COMBINATIONS)}")
    
    if best_result:
        print(f"\n✅ BEST RESULT:")
        print(f"  out_ch: {best_result['out_ch']}")
        print(f"  num_heads: {best_result['num_heads']}")
        print(f"  mid_ch: {best_result['mid_ch']}")
        print(f"  Best Valid AUROC: {best_result['best_valid_auroc']:.4f} (Epoch {best_result['best_epoch']})")
        print(f"  Test AUROC: {best_result['test_auroc']:.4f}")
        print(f"  Test AUPRC: {best_result['test_auprc']:.4f}")
        print(f"  Test Macro F1: {best_result['test_macro_f1']:.4f}")
        print(f"  Test Accuracy: {best_result['test_accuracy']:.4f}")
    else:
        print(f"\n❌ No successful trials!")
    
    print(f"\nResults saved to: {search_dir}")
    print(f"{'='*80}\n")
    
    return results_list, best_result, search_dir


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 그리드 서치 실행
    results_list, best_result, search_dir = run_gridsearch(device)
    
    # 최적 파라미터 출력
    if best_result:
        print(f"\n✨ NEXT STEP: Run main_autoexp.py with optimal parameters:")
        print(f"  out_ch: {best_result['out_ch']}")
        print(f"  num_heads: {best_result['num_heads']}")
        print(f"  mid_ch: {best_result['mid_ch']}")
        print(f"\nUpdate MODEL_CONFIG in main_autoexp.py before running!")
    else:
        print(f"\n⚠️  Could not find best parameters!")
