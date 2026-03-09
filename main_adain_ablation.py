# main_adain_ablation.py - ResU-Dense AdaIN Ablation 실험
#
# 실험 종류:
#   [1] RD_inter   : ResU-Dense CrossAttention, 표준 inter-patient (AdaIN 없음)
#   [2] RD_AdaIN   : ResU-Dense + Patient-Specific AdaIN (3-phase: DS1→DS2)
#
# 사용법: python main_adain_ablation.py

import os
import time
import copy
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import (set_seed,
                   load_or_extract_data,
                   intra_patient_split,
                   FocalLoss)
from model import get_model, PatientClassifier
from dataloader import ECGDataset
from train import train_one_epoch, validate
from test import evaluate, calculate_metrics, print_metrics, save_results_excel, save_confusion_matrix
from logger import calculate_auprc, calculate_auroc

# =============================================================================
# 전역 설정
# =============================================================================

USE_FOCAL_LOSS   = False
SELECTION_METRIC = "gmean"   # "macro_auroc" | "macro_auprc" | "gmean"
CALIB_MINUTES    = 1         # Phase 2/3 AdaIN용 DS2 calib 구간 (분)

DATA_PATH    = './data/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_PATH  = './ablation_results/'
BATCH_SIZE   = 1024
EPOCHS       = 50
LR           = 0.0001
WEIGHT_DECAY = 1e-3
SEED         = 1234
POLY1_EPS    = 0.0
POLY2_EPS    = 0.0
CLASSES      = ['N', 'S', 'V', 'F']

VALID_LEADS = ['MLII', 'V1', 'V2', 'V4', 'V5']
OUT_LEN     = 720

RR_FEATURE_OPTION = "opt3"
RR_FEATURE_DIMS   = {"opt1": 7, "opt2": 38, "opt3": 7, "opt4": 7}

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

# =============================================================================
# ResU-Dense 모델 설정
# =============================================================================

MODEL_CONFIG = {
    'in_channels': 1, 'out_ch': 180, 'mid_ch': 30, 'num_heads': 9,
    'n_rr': RR_FEATURE_DIMS[RR_FEATURE_OPTION],
}

# PatientClassifier 입력 채널 = ResU-Dense out_ch
PATCLS_IN_CH = MODEL_CONFIG['out_ch']   # 180

# =============================================================================
# 실험 정의
# =============================================================================

# (exp_name, model_type, exp_kind)
EXPERIMENTS = [
    ('RD_inter',  'cross_attention',  'inter'),   # AdaIN 없음 (기준)
    ('RD_AdaIN',  'resu_dense_adain', 'adain'),   # AdaIN 3-phase
]

# =============================================================================
# 유틸리티
# =============================================================================

def get_selection_val(metrics: dict, sel_metric: str) -> float:
    if sel_metric == 'gmean':
        recall = metrics.get('per_class_recall', np.zeros(4))
        return float(np.prod(np.maximum(recall, 1e-6)) ** (1.0 / len(recall)))
    return float(metrics.get(sel_metric, 0.0))


def make_criterion(labels_arr=None, device='cpu'):
    if USE_FOCAL_LOSS:
        counts = np.bincount(labels_arr, minlength=4).tolist() if labels_arr is not None else None
        return FocalLoss(class_counts=counts, device=device)
    return None


def create_exp_dir(exp_name: str) -> str:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_PATH, f'{exp_name}_{ts}')
    os.makedirs(os.path.join(exp_dir, 'best_weights'), exist_ok=True)
    return exp_dir


def make_dataloader(dataset, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    def worker_init(wid):
        np.random.seed(SEED + wid)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True, worker_init_fn=worker_init)


def load_resu_data(record_list: list, split_name: str):
    return load_or_extract_data(
        record_list=record_list, base_path=DATA_PATH, valid_leads=VALID_LEADS,
        out_len=OUT_LEN, split_name=split_name
    )


def eval_and_metrics(model, loader, device) -> dict:
    y_pred, y_true, eval_results = evaluate(model, loader, device)
    metrics = calculate_metrics(np.array(y_true), np.array(y_pred))
    try:
        y_probs = np.stack([r['all_probs'] for r in eval_results])
        n_cls = y_probs.shape[1]
        macro_auprc, weighted_auprc = calculate_auprc(np.array(y_true), y_probs, n_cls)
        macro_auroc, weighted_auroc = calculate_auroc(np.array(y_true), y_probs, n_cls)
        metrics.update({
            'macro_auroc': macro_auroc, 'weighted_auroc': weighted_auroc,
            'macro_auprc': macro_auprc, 'weighted_auprc': weighted_auprc,
        })
    except Exception as e:
        print(f"  Warning: AUROC/AUPRC calculation failed - {e}")
    return metrics


def _split_ds2_calib_eval(data, labels, rr, pids, sids, calib_minutes: int, fs: int = 360):
    calib_samples = calib_minutes * 60 * fs
    calib_mask = sids < calib_samples
    eval_mask  = ~calib_mask
    return (data[calib_mask], labels[calib_mask], rr[calib_mask],
            pids[calib_mask],  sids[calib_mask],
            data[eval_mask],  labels[eval_mask],  rr[eval_mask],
            pids[eval_mask],   sids[eval_mask])


# =============================================================================
# [1] Inter-patient (표준, AdaIN 없음)
# =============================================================================

def run_experiment(exp_name: str, model_type: str, device) -> tuple:
    print(f"\n{'='*80}")
    print(f"[Inter] {exp_name}  |  Model: {model_type}")
    print(f"{'='*80}")

    set_seed(SEED)
    exp_dir = create_exp_dir(exp_name)

    train_data, train_labels, train_rr, train_pid, train_sid = \
        load_resu_data(DS1_TRAIN_SPLIT, "RD_Train")
    valid_data, valid_labels, valid_rr, valid_pid, valid_sid = \
        load_resu_data(DS1_VALID_SPLIT, "RD_Valid")
    test_data, test_labels, test_rr, test_pid, test_sid = \
        load_resu_data(DS2_TEST, "RD_Test")

    train_loader = make_dataloader(
        ECGDataset(train_data, train_rr, train_labels, train_pid, train_sid), shuffle=True)
    valid_loader = make_dataloader(
        ECGDataset(valid_data, valid_rr, valid_labels, valid_pid, valid_sid), shuffle=False)
    test_loader  = make_dataloader(
        ECGDataset(test_data,  test_rr,  test_labels,  test_pid,  test_sid),  shuffle=False)

    print(f"  Train: {len(train_labels):,} | Valid: {len(valid_labels):,} | Test: {len(test_labels):,}")
    print(f"  Train dist: {dict(Counter(train_labels.tolist()))}")

    n_records = len(DS1_TRAIN_SPLIT) + len(DS1_VALID_SPLIT)
    model = get_model(exp_name=model_type, nOUT=len(CLASSES), n_pid=n_records,
                      **MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = make_criterion(train_labels, device)

    best = {'value': -1.0, 'epoch': 0, 'state_dict': None}

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, POLY1_EPS, POLY2_EPS, optimizer, device, criterion)
        _, valid_metrics, *_ = validate(model, valid_loader, POLY1_EPS, POLY2_EPS, device, criterion)
        sel_val = get_selection_val(valid_metrics, SELECTION_METRIC)
        if sel_val > best['value']:
            best = {'value': sel_val, 'epoch': epoch,
                    'state_dict': copy.deepcopy(model.state_dict())}
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{EPOCHS} — Valid {SELECTION_METRIC}: {sel_val:.4f}")
        scheduler.step()

    save_path = os.path.join(exp_dir, 'best_weights', f'best_{exp_name}.pth')
    torch.save({'model_state_dict': best['state_dict'], 'epoch': best['epoch'],
                SELECTION_METRIC: best['value']}, save_path)
    print(f"  Best epoch {best['epoch']} ({SELECTION_METRIC}={best['value']:.4f})")

    model.load_state_dict(best['state_dict'])
    model.eval()
    metrics = eval_and_metrics(model, test_loader, device)
    print_metrics(metrics, CLASSES)
    save_results_excel(metrics, CLASSES, os.path.join(exp_dir, f'results_{exp_name}.xlsx'))
    save_confusion_matrix(metrics['confusion_matrix'], CLASSES,
                          os.path.join(exp_dir, f'cm_{exp_name}.png'))

    return metrics, exp_dir


# =============================================================================
# [2] ResU-Dense AdaIN 3-phase
# =============================================================================

def run_experiment_adain_resu(exp_name: str, device) -> tuple:
    """
    3-phase AdaIN training (ResU-Dense):
        Phase 1: DS1 → ResU_Dense_AdaIN 기본 학습 (p_vec=None)
        Phase 2: DS2 calib → PatientClassifier 학습 (backbone freeze)
        Phase 3: DS2 calib → AdaIN fine-tune (앞단 freeze, fixed p_vec)
        Test:    DS2 eval → 평가
    """
    print(f"\n{'='*80}")
    print(f"[ResU-Dense AdaIN 3-Phase] {exp_name}")
    print(f"{'='*80}")

    set_seed(SEED)
    exp_dir = create_exp_dir(exp_name)

    # ── 데이터 로드 ───────────────────────────────────────────────────────
    ds1_data, ds1_labels, ds1_rr, ds1_pids, ds1_sids = \
        load_resu_data(DS1_FULL, "RD_AdaIN_DS1")
    ds2_data, ds2_labels, ds2_rr, ds2_pids, ds2_sids = \
        load_resu_data(DS2_TEST, "RD_AdaIN_DS2")

    (calib_data, calib_labels, calib_rr, calib_pids, calib_sids,
     eval_data,  eval_labels,  eval_rr,  eval_pids,  eval_sids) = \
        _split_ds2_calib_eval(ds2_data, ds2_labels, ds2_rr, ds2_pids, ds2_sids, CALIB_MINUTES)

    print(f"  DS1 train: {len(ds1_labels):,}  |  DS2 calib: {len(calib_labels):,}"
          f"  |  DS2 eval: {len(eval_labels):,}")

    # DS2 patient_id → 0-based mapping
    unique_ds2_pids = sorted(np.unique(ds2_pids).tolist())
    ds2_pid_to_local = {pid: i for i, pid in enumerate(unique_ds2_pids)}
    n_patients_ds2 = len(unique_ds2_pids)

    ds1_loader = make_dataloader(
        ECGDataset(ds1_data,   ds1_rr,   ds1_labels,   ds1_pids,   ds1_sids),   shuffle=True)
    calib_loader = make_dataloader(
        ECGDataset(calib_data, calib_rr, calib_labels, calib_pids, calib_sids), shuffle=True)
    calib_loader_ns = make_dataloader(
        ECGDataset(calib_data, calib_rr, calib_labels, calib_pids, calib_sids), shuffle=False)
    eval_loader = make_dataloader(
        ECGDataset(eval_data,  eval_rr,  eval_labels,  eval_pids,  eval_sids),  shuffle=False)

    ce = torch.nn.CrossEntropyLoss()

    # ── Phase 1: DS1 기본 학습 (p_vec=None) ─────────────────────────────
    print("\n[Phase 1] DS1 기본 학습...")
    model = get_model(exp_name='resu_dense_adain', nOUT=len(CLASSES),
                      n_pid=len(np.unique(ds1_pids)), **MODEL_CONFIG).to(device)
    optimizer1 = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=EPOCHS)
    criterion1 = make_criterion(ds1_labels, device) or ce

    best_p1 = {'value': -1.0, 'epoch': 0, 'state_dict': None}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in ds1_loader:
            ecg, rr_feat, lbls, _pids, _idx = batch
            ecg = ecg.to(device); rr_feat = rr_feat.to(device); lbls = lbls.to(device)
            optimizer1.zero_grad()
            logits, _ = model(ecg, rr_feat, p_vec=None)
            loss = criterion1(logits, lbls)
            loss.backward()
            optimizer1.step()
            total_loss += loss.item()

        scheduler1.step()

        if epoch % 5 == 0 or epoch == EPOCHS:
            _, val_metrics, *_ = validate(model, calib_loader_ns, POLY1_EPS, POLY2_EPS,
                                          device, criterion1)
            sel_val = get_selection_val(val_metrics, SELECTION_METRIC)
            if sel_val > best_p1['value']:
                best_p1 = {'value': sel_val, 'epoch': epoch,
                           'state_dict': copy.deepcopy(model.state_dict())}
            print(f"  Phase1 Epoch {epoch}/{EPOCHS} — Loss: {total_loss/len(ds1_loader):.4f}, "
                  f"Calib {SELECTION_METRIC}: {sel_val:.4f}")

    model.load_state_dict(best_p1['state_dict'])
    torch.save({'model_state_dict': best_p1['state_dict'], 'epoch': best_p1['epoch']},
               os.path.join(exp_dir, 'best_weights', f'phase1_{exp_name}.pth'))
    print(f"  Phase 1 done: best epoch {best_p1['epoch']}")

    # ── Phase 2: PatientClassifier 학습 (backbone freeze) ───────────────
    print("\n[Phase 2] PatientClassifier 학습 (DS2 calib, backbone frozen)...")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # ResU-Dense out_ch = 180
    pat_cls = PatientClassifier(in_ch=PATCLS_IN_CH, gru_hidden=64, hidden_dim=64,
                                n_patients=n_patients_ds2).to(device)
    optimizer2 = torch.optim.Adam(pat_cls.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=30)

    for epoch in range(1, 31):
        pat_cls.train()
        total_loss = 0.0
        for batch in calib_loader:
            ecg, rr_feat, lbls, pat_ids, _idx = batch
            ecg = ecg.to(device)
            optimizer2.zero_grad()
            mid_feat = model.get_mid_feat(ecg)   # (B, 180, L'), no_grad inside
            local_pids = torch.tensor(
                [ds2_pid_to_local[int(p)] for p in pat_ids.numpy()],
                dtype=torch.long, device=device)
            pat_logits, _ = pat_cls(mid_feat)
            loss = ce(pat_logits, local_pids)
            loss.backward()
            optimizer2.step()
            total_loss += loss.item()

        scheduler2.step()
        if epoch % 10 == 0:
            print(f"  Phase2 Epoch {epoch}/30 — Loss: {total_loss/len(calib_loader):.4f}")

    torch.save({'pat_cls_state': pat_cls.state_dict()},
               os.path.join(exp_dir, 'best_weights', f'phase2_{exp_name}.pth'))
    print("  Phase 2 done")

    # ── p_vec 계산 (각 DS2 환자의 평균 embedding) ────────────────────────
    pat_cls.eval()
    model.eval()
    patient_pvecs = {}

    with torch.no_grad():
        for batch in calib_loader_ns:
            ecg, rr_feat, lbls, pat_ids, _idx = batch
            ecg = ecg.to(device)
            mid_feat = model.get_mid_feat(ecg)
            _, p_vec = pat_cls(mid_feat)          # (B, 64)
            for b_i, pid in enumerate(pat_ids.numpy()):
                pid = int(pid)
                if pid not in patient_pvecs:
                    patient_pvecs[pid] = []
                patient_pvecs[pid].append(p_vec[b_i].detach().cpu())

    for pid in patient_pvecs:
        patient_pvecs[pid] = torch.stack(patient_pvecs[pid]).mean(0)  # (64,)

    # ── Phase 3: AdaIN fine-tune ─────────────────────────────────────────
    print("\n[Phase 3] AdaIN fine-tune (DS2 calib)...")
    # 앞단 freeze: conv, bn, rub_0
    FREEZE_PREFIXES = ('conv', 'bn', 'rub_0')
    for name, param in model.named_parameters():
        frozen = any(name.startswith(pfx) for pfx in FREEZE_PREFIXES)
        param.requires_grad_(not frozen)

    optimizer3 = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5)
    criterion3 = make_criterion(calib_labels, device) or ce

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0
        for batch in calib_loader:
            ecg, rr_feat, lbls, pat_ids, _idx = batch
            ecg = ecg.to(device); rr_feat = rr_feat.to(device); lbls = lbls.to(device)
            pvec_batch = torch.stack([
                patient_pvecs.get(int(pid), torch.zeros(64))
                for pid in pat_ids.numpy()
            ]).to(device)
            optimizer3.zero_grad()
            logits, _ = model(ecg, rr_feat, p_vec=pvec_batch)
            loss = criterion3(logits, lbls)
            loss.backward()
            optimizer3.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Phase3 Epoch {epoch}/30 — Loss: {total_loss/len(calib_loader):.4f}")

    torch.save({'model_state_dict': model.state_dict()},
               os.path.join(exp_dir, 'best_weights', f'phase3_{exp_name}.pth'))
    print("  Phase 3 done")

    # ── 최종 테스트 (DS2 eval) ────────────────────────────────────────────
    print("\n[Test] DS2 eval...")
    model.eval()

    y_pred_all, y_true_all, probs_all = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            ecg, rr_feat, lbls, pat_ids, _idx = batch
            ecg = ecg.to(device); rr_feat = rr_feat.to(device)
            pvec_batch = torch.stack([
                patient_pvecs.get(int(pid), torch.zeros(64))
                for pid in pat_ids.numpy()
            ]).to(device)
            logits, _ = model(ecg, rr_feat, p_vec=pvec_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_pred_all.extend(preds.cpu().numpy().tolist())
            y_true_all.extend(lbls.numpy().tolist())
            probs_all.append(probs.cpu().numpy())

    y_pred  = np.array(y_pred_all)
    y_true  = np.array(y_true_all)
    y_probs = np.concatenate(probs_all, axis=0)

    metrics = calculate_metrics(y_true, y_pred)
    try:
        n_cls = y_probs.shape[1]
        macro_auprc, weighted_auprc = calculate_auprc(y_true, y_probs, n_cls)
        macro_auroc, weighted_auroc = calculate_auroc(y_true, y_probs, n_cls)
        metrics.update({'macro_auroc': macro_auroc, 'weighted_auroc': weighted_auroc,
                        'macro_auprc': macro_auprc, 'weighted_auprc': weighted_auprc})
    except Exception as e:
        print(f"  Warning: AUROC/AUPRC failed - {e}")

    print_metrics(metrics, CLASSES)
    save_results_excel(metrics, CLASSES, os.path.join(exp_dir, f'results_{exp_name}.xlsx'))
    save_confusion_matrix(metrics['confusion_matrix'], CLASSES,
                          os.path.join(exp_dir, f'cm_{exp_name}.png'))

    return metrics, exp_dir


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("ResU-Dense AdaIN Ablation Study")
    print("  [1] RD_inter  : standard inter-patient (no AdaIN)")
    print("  [2] RD_AdaIN  : inter-patient + Patient-Specific AdaIN (3-phase)")
    print("=" * 80)
    print(f"Device:           {device}")
    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"USE_FOCAL_LOSS:   {USE_FOCAL_LOSS}")
    print(f"SELECTION_METRIC: {SELECTION_METRIC}")
    print(f"CALIB_MINUTES:    {CALIB_MINUTES}")
    print("=" * 80)

    all_results = {}
    total_start = time.time()

    for exp_name, model_type, exp_kind in EXPERIMENTS:
        try:
            if exp_kind == 'adain':
                metrics, exp_dir = run_experiment_adain_resu(exp_name, device)
            else:  # 'inter'
                metrics, exp_dir = run_experiment(exp_name, model_type, device)

            all_results[exp_name] = metrics
        except Exception as e:
            print(f"\n[ERROR] {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    total_time = (time.time() - total_start) / 60
    print(f"\nTotal time: {total_time:.1f} min")

    # 최종 요약
    print("\n" + "=" * 80)
    print("Ablation Results Summary")
    print("=" * 80)
    for exp_name, m in all_results.items():
        auroc_str = f", AUROC={m['macro_auroc']:.4f}" if 'macro_auroc' in m else ""
        auprc_str = f", AUPRC={m['macro_auprc']:.4f}" if 'macro_auprc' in m else ""
        print(f"  {exp_name:15s}: F1={m.get('macro_f1', 0):.4f}, "
              f"G-Mean={m.get('gmean', 0):.4f}{auroc_str}{auprc_str}")

    if len(all_results) == 2:
        rd = all_results.get('RD_inter', {})
        ada = all_results.get('RD_AdaIN', {})
        print("\n[AdaIN 효과]")
        print(f"  F1   : {rd.get('macro_f1', 0):.4f} → {ada.get('macro_f1', 0):.4f}"
              f"  ({ada.get('macro_f1', 0) - rd.get('macro_f1', 0):+.4f})")
        print(f"  GMean: {rd.get('gmean', 0):.4f} → {ada.get('gmean', 0):.4f}"
              f"  ({ada.get('gmean', 0) - rd.get('gmean', 0):+.4f})")
        if 'macro_auroc' in rd and 'macro_auroc' in ada:
            print(f"  AUROC: {rd['macro_auroc']:.4f} → {ada['macro_auroc']:.4f}"
                  f"  ({ada['macro_auroc'] - rd['macro_auroc']:+.4f})")

    print(f"\nResults in: {OUTPUT_PATH}")
    print("=" * 80)
