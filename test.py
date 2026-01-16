import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score)

def evaluate(
    model,
    test_loader,
    device,
    rr_ablation=False,
    target_class=None,      # SVEB
    centroid_dict=None,     # {class_id: np.array(feature_dim)}
    max_samples=None      
):

    model.eval()
    eval_results = []
    y_pred, y_true, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            ecg_inputs, rr_features, labels, patient_id, idx = batch

            ecg_inputs = ecg_inputs.to(device)

            if rr_ablation:
                rr_features = torch.zeros_like(rr_features).to(device)
            else:
                rr_features = rr_features.to(device)

            logits, features = model(ecg_inputs, rr_features, rr_remove_ablation=rr_ablation)
            probs = F.softmax(logits, dim=1)

            top2 = torch.topk(probs, k=2, dim=1)

            pred = top2.indices[:, 0] #가장 큰 확률임 (argmax역할)
            margin = top2.values[:, 0] - top2.values[:, 1]

            for i in range(len(labels)):
                y_t = int(labels[i])
                y_p = int(pred[i])
                prob = probs[i]
                # confusion type
                if y_t == 1 and y_p == 1:
                    conf_type = "TP"
                elif y_t == 1 and y_p != 1:
                    conf_type = "FN"
                elif y_t != 1 and y_p == 1:
                    conf_type = "FP"
                else:
                    conf_type = "TN"

                item = {
                    "dataset_idx": int(idx[i]),
                    "y_true": y_t,
                    "y_pred": y_p,
                    "all_probs":prob.detach().cpu().numpy(),
                    "confidence": float(top2.values[i, 0]),
                    "margin": float(margin[i]),
                    "patient_id": int(patient_id[i]),
                    "conf_type": conf_type,
                }
                y_pred.append(y_p)
                y_true.append(y_t)
                all_probs.append(probs.detach().cpu().numpy())

                if target_class is not None and centroid_dict is not None:
                    feat = features[i].detach().cpu().numpy()
                    centroid = centroid_dict[target_class]
                    dist = np.linalg.norm(feat - centroid)
                    item["centroid_dist"] = dist

                eval_results.append(item)

    if target_class is not None and centroid_dict is not None and max_samples is not None:
        eval_results = sorted(
            eval_results,
            key=lambda x: x.get("centroid_dist", np.inf)
        )[:max_samples]

    return y_pred, y_true, eval_results

#======================================================================================

def filter_ambiguous_samples(eval_results,
                             y_true=None,
                             y_pred=None,
                             margin_thresh=0.2,
                             allowed_preds=None):
    """
    Generic ambiguous sample filter
    """
    selected = []

    for r in eval_results:
        if y_true is not None and r["y_true"] != y_true:
            continue

        if y_pred is not None and r["y_pred"] != y_pred:
            continue

        if allowed_preds is not None and r["y_pred"] not in allowed_preds:
            continue

        if r["margin"] >= margin_thresh:
            continue

        selected.append(r)

    return selected

def extract_ecg_signal(dataset, idx):
    ecg = dataset.dataset[idx][0]
    return ecg.squeeze().numpy()

def plot_and_save_ecg(signal, title, save_path):
    plt.figure(figsize=(8, 3))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import numpy as np


def format_probs_bold_max(probs, precision=3):
    """
    probs: np.ndarray or list, shape (C,) or (1, C) or (C, 1)
    return: LaTeX-formatted string with max prob in bold
    """
    probs = np.asarray(probs).squeeze()  
    assert probs.ndim == 1, f"Expected 1D probs, got shape {probs.shape}"

    max_idx = np.argmax(probs)

    formatted = []
    for i, p in enumerate(probs):
        p = float(p)  # ★ double safety
        p_str = f"{p:.{precision}f}"
        if i == max_idx:
            formatted.append(rf"\mathbf{{{p_str}}}")
        else:
            formatted.append(p_str)

    return r"$[" + ",\ ".join(formatted) + r"]$"

#======================================================================================

def get_ambiguous_sveb(eval_results, s_class_id, n_class_id, margin_thresh=0.2):
    return filter_ambiguous_samples(
        eval_results,
        y_true=s_class_id,
        margin_thresh=margin_thresh,
        allowed_preds=[s_class_id, n_class_id]
    )


def get_ambiguous_correct_sveb(eval_results, s_class_id, margin_thresh=0.2):
    return filter_ambiguous_samples(
        eval_results,
        y_true=s_class_id,
        y_pred=s_class_id,
        margin_thresh=margin_thresh
    )


def get_ambiguous_n_as_s(eval_results, s_class_id, n_class_id, margin_thresh=0.2):
    return filter_ambiguous_samples(
        eval_results,
        y_true=n_class_id,
        y_pred=s_class_id,
        margin_thresh=margin_thresh
    )

#======================================================================================

def plot_ambiguous_sveb(result_path, eval_results, dataset, classes,
                        s_class_id, n_class_id,
                        margin_thresh=0.2,
                        k=10):

    amb = get_ambiguous_sveb(
        eval_results, s_class_id, n_class_id, margin_thresh
    )[:k]

    for r in amb:
        idx = r["dataset_idx"]
        signal = extract_ecg_signal(dataset, idx)
        probs_str = format_probs_bold_max(r["all_probs"])
        title = (
            f"t=S | p={classes[r['y_pred']]} | "
            f"m={r['margin']:.3f} | Pr={probs_str}"
        )

        plot_and_save_ecg(signal, title, f"{result_path}/{idx}.png")

def plot_ambiguous_correct_sveb(result_path,
                                eval_results,
                                dataset,
                                s_class_id,
                                margin_thresh=0.2,
                                k=10):

    amb = get_ambiguous_correct_sveb(
        eval_results, s_class_id, margin_thresh
    )[:k]

    for r in amb:
        idx = r["dataset_idx"]
        signal = extract_ecg_signal(dataset, idx)
        probs_str = format_probs_bold_max(r["all_probs"])
        title = (
            f"true=S | pred=S | margin={r['margin']:.3f} | "
            f"Probs={r['all_probs']} | Probs={probs_str}"
        )

        plot_and_save_ecg(signal, title, f"{result_path}/TP_{idx}.png")

def plot_ambiguous_n_as_s(result_path, eval_results, dataset, classes,
                          s_class_id, n_class_id,
                          margin_thresh=0.2,
                          k=10):

    amb = get_ambiguous_n_as_s(
        eval_results, s_class_id, n_class_id, margin_thresh
    )[:k]

    for r in amb:
        idx = r["dataset_idx"]
        signal = extract_ecg_signal(dataset, idx)
        probs_str = format_probs_bold_max(r["all_probs"])
        title = (
            f"true=N | pred={classes[r['y_pred']]} | "
            f"margin={r['margin']:.3f}"
            f"| Probs={probs_str}"
        )

        plot_and_save_ecg(signal, title, f"{result_path}/N_as_S_{idx}.png")

    print(f"Saved {len(amb)} ambiguous N→S samples to {result_path}")

#======================================================================================

# 양방향 ambiguous 샘플을 모두 시각화하는 통합 함수
def plot_bidirectional_ambiguous(result_path, eval_results, dataset, classes,
                                  s_class_id, n_class_id,
                                  margin_thresh=0.2,
                                  k=10):

    import os
    os.makedirs(result_path, exist_ok=True)
    
    # S를 N으로 잘못 예측한 경우
    print("Plotting ambiguous S→N samples...")
    plot_ambiguous_sveb(result_path, eval_results, dataset, classes,
                       s_class_id, n_class_id, margin_thresh, k)
    
    # N을 S로 잘못 예측한 경우
    print("Plotting ambiguous N→S samples...")
    plot_ambiguous_n_as_s(result_path, eval_results, dataset, classes,
                         s_class_id, n_class_id, margin_thresh, k)

#======================================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(cm)

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)  # sensitivity
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Per-class specificity 계산 (One-vs-Rest)
    per_class_specificity = np.zeros(n_classes)
    for i in range(n_classes):
        # TN: 실제 i가 아니고 예측도 i가 아닌 경우
        # FP: 실제 i가 아닌데 i로 예측한 경우
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        per_class_specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Per-class accuracy (각 클래스에 대한 정확도)
    per_class_accuracy = np.zeros(n_classes)
    for i in range(n_classes):
        tp = cm[i, i]
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        total = np.sum(cm)
        per_class_accuracy[i] = (tp + tn) / total if total > 0 else 0.0

    # Macro metrics
    macro_prec = per_class_precision.mean()
    macro_recall = per_class_recall.mean()  # macro sensitivity
    macro_specificity = per_class_specificity.mean()
    macro_f1 = per_class_f1.mean()
    macro_accuracy = per_class_accuracy.mean()

    # Overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)

    # Weighted metrics
    class_counts = np.bincount(y_true, minlength=n_classes)
    class_weights = class_counts / len(y_true)

    weighted_prec = np.sum(per_class_precision * class_weights)
    weighted_recall = np.sum(per_class_recall * class_weights)  # weighted sensitivity
    weighted_specificity = np.sum(per_class_specificity * class_weights)
    weighted_f1 = np.sum(per_class_f1 * class_weights)
    weighted_accuracy = np.sum(per_class_accuracy * class_weights)

    return {
        'confusion_matrix': cm,
        'overall_accuracy': overall_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_specificity': per_class_specificity,
        'per_class_accuracy': per_class_accuracy,
        'per_class_f1': per_class_f1,
        'macro_prec': macro_prec,
        'macro_recall': macro_recall,
        'macro_specificity': macro_specificity,
        'macro_accuracy': macro_accuracy,
        'macro_f1': macro_f1,
        'weighted_prec': weighted_prec,
        'weighted_recall': weighted_recall,
        'weighted_specificity': weighted_specificity,
        'weighted_accuracy': weighted_accuracy,
        'weighted_f1': weighted_f1,
    }


def print_metrics(metrics: dict, classes: list) -> None:
    print(f"\n{'='*80}")
    print(f"Results")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"\n{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for i, cls in enumerate(classes):
        print(f"{cls:<10} {metrics['per_class_precision'][i]:<12.4f} "
              f"{metrics['per_class_recall'][i]:<12.4f} "
              f"{metrics['per_class_f1'][i]:<12.4f}")
    
    print("\n" + "-" * 80)
    print(f"{'Macro Avg':<10} {metrics['macro_prec']:<12.4f} "
          f"{metrics['macro_recall']:<12.4f} "
          f"{metrics['macro_f1']:<12.4f}")
    
    print(f"{'Weighted':<10} {metrics['weighted_prec']:<12.4f} "
          f"{metrics['weighted_recall']:<12.4f} "
          f"{metrics['weighted_f1']:<12.4f}")
    print("=" * 80)


def save_results_excel(metrics: dict, 
                       classes: list, 
                       save_path: str, 
                       lambda_value: float = None) -> None:
    """
    4개의 시트를 가진 Excel 파일로 결과 저장:
    1. Macro: Accuracy, Sensitivity, Specificity, Precision, F1
    2. Weighted: Accuracy, Sensitivity, Specificity, Precision, F1
    3. Per_Class: 각 클래스(N, S, V, F)별 5가지 메트릭
    4. Confusion_Matrix: 혼동 행렬
    """
    
    lambda_str = lambda_value if lambda_value is not None else 'N/A'
    
    # Sheet 1: Macro metrics
    macro_data = [{
        'Lambda': lambda_str,
        'Accuracy': metrics['macro_accuracy'],
        'Sensitivity': metrics['macro_recall'],
        'Specificity': metrics['macro_specificity'],
        'Precision': metrics['macro_prec'],
        'F1-Score': metrics['macro_f1']
    }]
    df_macro = pd.DataFrame(macro_data)
    
    # Sheet 2: Weighted metrics
    weighted_data = [{
        'Lambda': lambda_str,
        'Accuracy': metrics['weighted_accuracy'],
        'Sensitivity': metrics['weighted_recall'],
        'Specificity': metrics['weighted_specificity'],
        'Precision': metrics['weighted_prec'],
        'F1-Score': metrics['weighted_f1']
    }]
    df_weighted = pd.DataFrame(weighted_data)
    
    # Sheet 3: Per-class metrics
    per_class_data = []
    for i, cls in enumerate(classes):
        per_class_data.append({
            'Lambda': lambda_str,
            'Class': cls,
            'Accuracy': metrics['per_class_accuracy'][i],
            'Sensitivity': metrics['per_class_recall'][i],
            'Specificity': metrics['per_class_specificity'][i],
            'Precision': metrics['per_class_precision'][i],
            'F1-Score': metrics['per_class_f1'][i]
        })
    df_per_class = pd.DataFrame(per_class_data)
    
    # Sheet 4: Confusion Matrix
    cm = metrics['confusion_matrix']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    df_cm.index.name = 'True \\ Predicted'
    
    # Excel 저장
    # .xlsx 확장자 확인
    if not save_path.endswith('.xlsx'):
        save_path = save_path.replace('.csv', '.xlsx')
        if not save_path.endswith('.xlsx'):
            save_path += '.xlsx'
    
    # 기존 파일이 있으면 기존 데이터에 추가
    if os.path.exists(save_path):
        try:
            existing_macro = pd.read_excel(save_path, sheet_name='Macro')
            existing_weighted = pd.read_excel(save_path, sheet_name='Weighted')
            existing_per_class = pd.read_excel(save_path, sheet_name='Per_Class')
            
            df_macro = pd.concat([existing_macro, df_macro], ignore_index=True)
            df_weighted = pd.concat([existing_weighted, df_weighted], ignore_index=True)
            df_per_class = pd.concat([existing_per_class, df_per_class], ignore_index=True)
        except Exception:
            pass  # 새 파일로 생성
    
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df_macro.to_excel(writer, sheet_name='Macro', index=False)
        df_weighted.to_excel(writer, sheet_name='Weighted', index=False)
        df_per_class.to_excel(writer, sheet_name='Per_Class', index=False)
        df_cm.to_excel(writer, sheet_name='Confusion_Matrix')
    
    print(f"✓ Results saved: {save_path}")


def save_confusion_matrix(cm: np.ndarray, 
                          classes: list, 
                          save_path: str, 
                          lambda_value: float = None) -> None:
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "DejaVu Sans"
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    title = 'Confusion Matrix'
    if lambda_value is not None:
        title += f' (λ={lambda_value})'
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")

#======================================================================================

def analyze_low_margin_sveb(eval_results, s_class_id, margin_thresh=0.2):
    """
    Low-margin SVEB 샘플의 비율과 FN rate 분석
    """
    sveb = [r for r in eval_results if r["y_true"] == s_class_id]
    low_margin = [r for r in sveb if r["margin"] < margin_thresh]

    fn_low_margin = [r for r in low_margin if r["conf_type"] == "FN"]

    stats = {
        "num_sveb": len(sveb),
        "num_low_margin": len(low_margin),
        "ratio_low_margin": len(low_margin) / max(len(sveb), 1),
        "fn_low_margin": len(fn_low_margin),
        "fn_rate_low_margin": len(fn_low_margin) / max(len(low_margin), 1)
    }
    return stats


def margin_statistics(eval_results, class_id):
    margins = np.array([
        r["margin"] for r in eval_results
        if r["y_true"] == class_id
    ])

    if len(margins) == 0:
        return {}

    return {
        "mean": margins.mean(),
        "median": np.median(margins),
        "std": margins.std(),
        "p10": np.percentile(margins, 10),
        "p25": np.percentile(margins, 25),
        "p50": np.percentile(margins, 50),
        "p75": np.percentile(margins, 75),
    }
