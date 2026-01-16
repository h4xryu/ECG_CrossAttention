# logger.py 업데이트

import os
os.environ["TENSORBOARD_NO_TF"] = "1"
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np


# =============================================================================
# ANSI Color Codes (터미널 지원 여부 자동 감지)
# =============================================================================
import sys

def _supports_color():
    """터미널이 ANSI 색상을 지원하는지 확인"""
    # Windows cmd는 기본적으로 지원 안함
    if sys.platform == 'win32':
        return False
    # TTY가 아니면 (파이프, 파일 리다이렉트 등) 지원 안함
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    return True

class Colors:
    _enabled = _supports_color()
    
    @classmethod
    def _c(cls, code):
        return code if cls._enabled else ''
    
    @property
    def RED(self):
        return '\033[91m' if self._enabled else ''
    
    @property
    def GREEN(self):
        return '\033[92m' if self._enabled else ''
    
    @property  
    def YELLOW(self):
        return '\033[93m' if self._enabled else ''
    
    @property
    def BLUE(self):
        return '\033[94m' if self._enabled else ''
    
    @property
    def MAGENTA(self):
        return '\033[95m' if self._enabled else ''
    
    @property
    def CYAN(self):
        return '\033[96m' if self._enabled else ''
    
    @property
    def WHITE(self):
        return '\033[97m' if self._enabled else ''
    
    @property
    def BOLD(self):
        return '\033[1m' if self._enabled else ''
    
    @property
    def RESET(self):
        return '\033[0m' if self._enabled else ''

# 싱글톤 인스턴스
Colors = Colors()


class TrainingLogger:
    
    def __init__(self, log_dir: str, lambda_val: float = None):
        suffix = f"_lambda_{lambda_val}" if lambda_val is not None else ""
        self.writer = SummaryWriter(log_dir=f'{log_dir}{suffix}')
        self.lambda_val = lambda_val
    
    def log_epoch(self, epoch: int, loss: float, metrics: dict, phase: str = 'train') -> None:
        """
        Log metrics for train or validation phase
        
        Args:
            phase: 'train' or 'valid'
        """
        prefix = f'{phase.capitalize()}'
        
        self.writer.add_scalar(f'{prefix}/Loss/total', loss, epoch)
        self.writer.add_scalar(f'{prefix}/Accuracy/overall', metrics['acc'], epoch)
        
        # Macro/Weighted metrics
        self.writer.add_scalar(f'{prefix}/Macro/precision', metrics['macro_precision'], epoch)
        self.writer.add_scalar(f'{prefix}/Macro/recall', metrics['macro_recall'], epoch)
        self.writer.add_scalar(f'{prefix}/Macro/f1', metrics['macro_f1'], epoch)
        self.writer.add_scalar(f'{prefix}/Weighted/precision', metrics['weighted_precision'], epoch)
        self.writer.add_scalar(f'{prefix}/Weighted/recall', metrics['weighted_recall'], epoch)
        self.writer.add_scalar(f'{prefix}/Weighted/f1', metrics['weighted_f1'], epoch)
        self.writer.add_scalar(f'{prefix}/AUPRC/macro', metrics['macro_auprc'], epoch)
        self.writer.add_scalar(f'{prefix}/AUPRC/weighted', metrics['weighted_auprc'], epoch)
        self.writer.add_scalar(f'{prefix}/AUROC/macro', metrics['macro_auroc'], epoch)
        self.writer.add_scalar(f'{prefix}/AUROC/weighted', metrics['weighted_auroc'], epoch)
        
        # Per-class metrics
        class_names = ['N', 'S', 'V', 'F']
        for i, name in enumerate(class_names):
            if i < len(metrics['per_class_acc']):
                self.writer.add_scalar(f'{prefix}/PerClass/{name}/accuracy', metrics['per_class_acc'][i], epoch)
                self.writer.add_scalar(f'{prefix}/PerClass/{name}/precision', metrics['per_class_precision'][i], epoch)
                self.writer.add_scalar(f'{prefix}/PerClass/{name}/recall', metrics['per_class_recall'][i], epoch)
                self.writer.add_scalar(f'{prefix}/PerClass/{name}/f1', metrics['per_class_f1'][i], epoch)
    
    def log_confidence(self, epoch: int, p_t_n, p_t_s, p_t_v, p_t_f, phase: str = 'train') -> None:
        """Log confidence statistics for each class"""
        prefix = f'{phase.capitalize()}/Confidence'
        
        # N class
        if len(p_t_n) > 0:
            self.writer.add_scalar(f"{prefix}/N/p_t_mean", p_t_n.mean(), epoch)
            self.writer.add_scalar(f"{prefix}/N/p_t_median", p_t_n.median(), epoch)
            self.writer.add_scalar(f"{prefix}/N/p_t_min", p_t_n.min(), epoch)
            self.writer.add_scalar(f"{prefix}/N/p_t_max", p_t_n.max(), epoch)
        
        # S class
        if len(p_t_s) > 0:
            self.writer.add_scalar(f"{prefix}/S/p_t_mean", p_t_s.mean(), epoch)
            self.writer.add_scalar(f"{prefix}/S/p_t_median", p_t_s.median(), epoch)
            self.writer.add_scalar(f"{prefix}/S/p_t_min", p_t_s.min(), epoch)
            self.writer.add_scalar(f"{prefix}/S/p_t_max", p_t_s.max(), epoch)
        
        # V class
        if len(p_t_v) > 0:
            self.writer.add_scalar(f"{prefix}/V/p_t_mean", p_t_v.mean(), epoch)
            self.writer.add_scalar(f"{prefix}/V/p_t_median", p_t_v.median(), epoch)
            self.writer.add_scalar(f"{prefix}/V/p_t_min", p_t_v.min(), epoch)
            self.writer.add_scalar(f"{prefix}/V/p_t_max", p_t_v.max(), epoch)
        
        # F class
        if len(p_t_f) > 0:
            self.writer.add_scalar(f"{prefix}/F/p_t_mean", p_t_f.mean(), epoch)
            self.writer.add_scalar(f"{prefix}/F/p_t_median", p_t_f.median(), epoch)
            self.writer.add_scalar(f"{prefix}/F/p_t_min", p_t_f.min(), epoch)
            self.writer.add_scalar(f"{prefix}/F/p_t_max", p_t_f.max(), epoch)
    
    def close(self):
        self.writer.close()


def print_epoch_header():
    print(f"\n{'='*120}")
    print(f"{'Epoch':<8} {'Loss':<12} {'Accuracy':<12} {'Learning Rate':<15}")
    print(f"{'='*120}")


def print_epoch_stats(epoch: int, loss: float, accuracy: float, lr: float, phase: str = 'Train'):
    """Print epoch stats with color: Train=RED, Valid=BLUE"""
    if phase == 'Train':
        color = Colors.RED
    else:
        color = Colors.BLUE
    
    line = f"[{phase}] Epoch {epoch:<4} Loss: {loss:<12.6f} Accuracy: {accuracy:<12.6f} LR: {lr:<15.8f}"
    separator = "=" * 120
    
    print(f"\n{color}{separator}{Colors.RESET}")
    print(f"{color}{Colors.BOLD}{line}{Colors.RESET}")
    print(f"{color}{separator}{Colors.RESET}")


def print_per_class_metrics(metrics: dict, classes: list = None, phase: str = 'Train') -> None:
    if classes is None:
        classes = ['N', 'S', 'V', 'F']
    
    print(f"\n{'='*120}")
    print(f"[{phase}] Per-Class Metrics:")
    print(f"{'='*120}")
    
    # Header
    print(f"{'Class':<8} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*120}")
    
    # Per-class values
    for i, cls in enumerate(classes):
        if i < len(metrics['per_class_acc']):
            print(f"{cls:<8} "
                  f"{metrics['per_class_acc'][i]:<12.4f} "
                  f"{metrics['per_class_precision'][i]:<12.4f} "
                  f"{metrics['per_class_recall'][i]:<12.4f} "
                  f"{metrics['per_class_f1'][i]:<12.4f}")
    
    print(f"{'-'*120}")
    
    # Macro averages
    print(f"{'Macro':<8} "
          f"{'':<12} "
          f"{metrics['macro_precision']:<12.4f} "
          f"{metrics['macro_recall']:<12.4f} "
          f"{metrics['macro_f1']:<12.4f}")
    
    # Weighted averages
    print(f"{'Weighted':<8} "
          f"{'':<12} "
          f"{metrics['weighted_precision']:<12.4f} "
          f"{metrics['weighted_recall']:<12.4f} "
          f"{metrics['weighted_f1']:<12.4f}")
    
    print(f"{'='*120}")
    
    # AUPRC and AUROC
    print(f"\nAUPRC  - Macro: {metrics['macro_auprc']:.4f}, Weighted: {metrics['weighted_auprc']:.4f}")
    print(f"AUROC  - Macro: {metrics['macro_auroc']:.4f}, Weighted: {metrics['weighted_auroc']:.4f}")
    print(f"{'='*120}")


def print_confidence_stats(p_t_n, p_t_s, p_t_v, p_t_f, phase: str = 'Train') -> None:
    """Print confidence statistics for each class"""
    print(f"\n[{phase}] Confidence Statistics (p_t):")
    print(f"{'='*120}")
    
    if len(p_t_n) > 0:
        print(f"N-class: mean={p_t_n.mean():.4f}, median={p_t_n.median():.4f}, "
              f"min={p_t_n.min():.4f}, max={p_t_n.max():.4f}")
    else:
        print(f"N-class: No samples")
    
    if len(p_t_s) > 0:
        print(f"S-class: mean={p_t_s.mean():.4f}, median={p_t_s.median():.4f}, "
              f"min={p_t_s.min():.4f}, max={p_t_s.max():.4f}")
    else:
        print(f"S-class: No samples")
    
    if len(p_t_v) > 0:
        print(f"V-class: mean={p_t_v.mean():.4f}, median={p_t_v.median():.4f}, "
              f"min={p_t_v.min():.4f}, max={p_t_v.max():.4f}")
    else:
        print(f"V-class: No samples")
    
    if len(p_t_f) > 0:
        print(f"F-class: mean={p_t_f.mean():.4f}, median={p_t_f.median():.4f}, "
              f"min={p_t_f.min():.4f}, max={p_t_f.max():.4f}")
    else:
        print(f"F-class: No samples")
    
    print(f"{'='*120}")


def print_epoch_time(epoch: int, elapsed_time: float):
    """Print epoch time in GREEN"""
    print(f"{Colors.GREEN}Time taken for epoch {epoch}: {elapsed_time:.2f} seconds{Colors.RESET}")


def calculate_auprc(y_true, y_probs, num_classes=5):
    """Calculate AUPRC for each class and return macro/weighted averages"""
    from sklearn.metrics import average_precision_score
    
    # One-hot encode y_true
    y_true_onehot = np.eye(num_classes)[y_true]
    
    # Per-class AUPRC
    auprc_per_class = []
    for i in range(num_classes):
        try:
            ap = average_precision_score(y_true_onehot[:, i], y_probs[:, i])
            auprc_per_class.append(ap)
        except:
            auprc_per_class.append(0.0)
    
    # Calculate weights (class frequencies)
    class_counts = np.bincount(y_true, minlength=num_classes)
    weights = class_counts / len(y_true)
    
    macro_auprc = np.mean(auprc_per_class)
    weighted_auprc = np.sum(np.array(auprc_per_class) * weights)
    
    return macro_auprc, weighted_auprc


def calculate_auroc(y_true, y_probs, num_classes=5):
    """Calculate AUROC for each class and return macro/weighted averages"""
    try:
        macro_auroc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        weighted_auroc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
    except:
        macro_auroc = 0.0
        weighted_auroc = 0.0
    
    return macro_auroc, weighted_auroc