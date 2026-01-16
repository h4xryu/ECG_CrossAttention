# train.py에 validation 함수 추가

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F


def train_one_epoch(model: nn.Module, train_loader, alpha1: float, alpha2: float,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple:
    model.train()
    total_loss = 0.0
    y_pred, y_true = [], []
    y_probs_all = []  # For AUPRC/AUROC
    
    p_t_n_all = []  # N 클래스
    p_t_s_all = []  # S 클래스
    p_t_v_all = []  # V 클래스
    p_t_f_all = []  # F 클래스
    
    for batch in train_loader:
        ecg_inputs, rr_features, labels, pids, _ = batch
        ecg_inputs = ecg_inputs.to(device)
        rr_features = rr_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(ecg_inputs, rr_features, rr_remove_ablation=False)
        probs = torch.softmax(logits, dim=1)
        p_t = probs[torch.arange(len(labels)), labels]  # (B,)
        
        # Store predictions and probabilities
        preds = torch.argmax(logits, dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_probs_all.append(probs.detach().cpu().numpy())
        
        # Collect p_t per class (클래스별 confidence 수집)
        n_mask = (labels == 0)
        if n_mask.any():
            p_t_n_all.append(p_t[n_mask].detach().cpu())
        
        s_mask = (labels == 1)
        if s_mask.any():
            p_t_s_all.append(p_t[s_mask].detach().cpu())
        
        v_mask = (labels == 2)
        if v_mask.any():
            p_t_v_all.append(p_t[v_mask].detach().cpu())
        
        f_mask = (labels == 3)
        if f_mask.any():
            p_t_f_all.append(p_t[f_mask].detach().cpu())
        
        # Loss calculation: CE + α₁(1-p_t) + α₂(1-p_t)²
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        poly_term = alpha1 * (1 - p_t) + alpha2 * (1 - p_t)**2
        loss = (ce_loss + poly_term).mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs_all = np.concatenate(y_probs_all, axis=0)
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # AUPRC and AUROC
    from logger import calculate_auprc, calculate_auroc
    num_classes = y_probs_all.shape[1]
    macro_auprc, weighted_auprc = calculate_auprc(y_true, y_probs_all, num_classes)
    macro_auroc, weighted_auroc = calculate_auroc(y_true, y_probs_all, num_classes)
    
    
    metrics = {
        'acc': accuracy,
        'per_class_acc': per_class_acc,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_auprc': macro_auprc,
        'weighted_auprc': weighted_auprc,
        'macro_auroc': macro_auroc,
        'weighted_auroc': weighted_auroc,
    }
    
    # Concatenate p_t arrays
    p_t_n_all = torch.cat(p_t_n_all, dim=0) if p_t_n_all else torch.empty(0)
    p_t_s_all = torch.cat(p_t_s_all, dim=0) if p_t_s_all else torch.empty(0)
    p_t_v_all = torch.cat(p_t_v_all, dim=0) if p_t_v_all else torch.empty(0)
    p_t_f_all = torch.cat(p_t_f_all, dim=0) if p_t_f_all else torch.empty(0)
    
    return (total_loss / len(train_loader), metrics, 
            p_t_n_all, p_t_s_all, p_t_v_all, p_t_f_all)


def validate(model: nn.Module, valid_loader, alpha1: float, alpha2: float, device: torch.device) -> tuple:
    """Validation function"""
    model.eval()
    total_loss = 0.0
    y_pred, y_true = [], []
    y_probs_all = []
    
    p_t_n_all = []
    p_t_s_all = []
    p_t_v_all = []
    p_t_f_all = []
    
    with torch.no_grad():
        for batch in valid_loader:
            ecg_inputs, rr_features, labels, pids, _ = batch
            ecg_inputs = ecg_inputs.to(device)
            rr_features = rr_features.to(device)
            labels = labels.to(device)
            
            logits, _ = model(ecg_inputs, rr_features, rr_remove_ablation=False)
            probs = torch.softmax(logits, dim=1)
            p_t = probs[torch.arange(len(labels)), labels]
            
            # Store predictions and probabilities
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            y_probs_all.append(probs.cpu().numpy())
            
            # Collect p_t per class
            n_mask = (labels == 0)
            if n_mask.any():
                p_t_n_all.append(p_t[n_mask].cpu())
            
            s_mask = (labels == 1)
            if s_mask.any():
                p_t_s_all.append(p_t[s_mask].cpu())
            
            v_mask = (labels == 2)
            if v_mask.any():
                p_t_v_all.append(p_t[v_mask].cpu())
            
            f_mask = (labels == 3)
            if f_mask.any():
                p_t_f_all.append(p_t[f_mask].cpu())
            
            # Loss calculation (for monitoring)
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            poly_term = alpha1 * (1 - p_t) + alpha2 * (1 - p_t)**2
            loss = (ce_loss + poly_term).mean()
            
            total_loss += loss.item()
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs_all = np.concatenate(y_probs_all, axis=0)
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # AUPRC and AUROC
    from logger import calculate_auprc, calculate_auroc
    num_classes = y_probs_all.shape[1]
    macro_auprc, weighted_auprc = calculate_auprc(y_true, y_probs_all, num_classes)
    macro_auroc, weighted_auroc = calculate_auroc(y_true, y_probs_all, num_classes)
    
    # Concatenate p_t arrays
    if len(p_t_n_all) > 0:
        p_t_n_all = torch.cat(p_t_n_all, dim=0)
    else:
        p_t_n_all = torch.empty(0)
    
    if len(p_t_s_all) > 0:
        p_t_s_all = torch.cat(p_t_s_all, dim=0)
    else:
        p_t_s_all = torch.empty(0)
    
    if len(p_t_v_all) > 0:
        p_t_v_all = torch.cat(p_t_v_all, dim=0)
    else:
        p_t_v_all = torch.empty(0)
    
    if len(p_t_f_all) > 0:
        p_t_f_all = torch.cat(p_t_f_all, dim=0)
    else:
        p_t_f_all = torch.empty(0)
    
    metrics = {
        'acc': accuracy,
        'per_class_acc': per_class_acc,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'macro_auprc': macro_auprc,
        'weighted_auprc': weighted_auprc,
        'macro_auroc': macro_auroc,
        'weighted_auroc': weighted_auroc,
    }
    
    return (total_loss / len(valid_loader), metrics, 
            p_t_n_all, p_t_s_all, p_t_v_all, p_t_f_all)


def save_model(model: nn.Module, 
               optimizer: torch.optim.Optimizer,
               epoch: int, 
               metrics: dict, 
               save_path: str) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved: {save_path}")


def load_model(model: nn.Module, 
               load_path: str, 
               optimizer: torch.optim.Optimizer = None,
               device: torch.device = None) -> tuple:
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Model loaded from: {load_path}")
    return model, optimizer, epoch, metrics