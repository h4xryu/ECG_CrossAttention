
import warnings
import torch

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_device_info():
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")



__version__ = '1.0.0'
__all__ = ['device', 'print_device_info', 'LIGHTNING_AVAILABLE']
