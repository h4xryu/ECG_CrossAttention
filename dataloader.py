
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from config import DATA_PATH, VALID_LEADS, OUT_LEN, BATCH_SIZE, DS1_TRAIN, DS2_TEST
from utils import extract_beats_and_rr_from_records, _build_x3

# =============================================================================
# PyTorch Dataset
# =============================================================================

class ECGDataset(Dataset):
    
    def __init__(self, ecg_data, rr_features, labels, patient_id, idx):
        self.ecg_data = torch.FloatTensor(ecg_data).unsqueeze(1)  # (N, 1, L)
        self.rr_features = torch.FloatTensor(rr_features)
        self.labels = torch.LongTensor(labels)
        self.patient_id = torch.LongTensor(patient_id)
        self.idx = idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.ecg_data[idx],
                self.rr_features[idx],
                self.labels[idx],
                self.patient_id[idx],
                idx) #샘플 그려볼거기 때문에

# =============================================================================
# Data Loading Functions 
# =============================================================================

def load_data(data_path: str = DATA_PATH, 
              valid_leads: list = VALID_LEADS,
              out_len: int = OUT_LEN,
              batch_size: int = BATCH_SIZE) -> tuple:

    # Extract data
    train_data, train_labels, train_rr, patient_id = extract_beats_and_rr_from_records(
        DS1_TRAIN, data_path, valid_leads, out_len, "Train"
    )

    test_data, test_labels, test_rr, patient_id_test = extract_beats_and_rr_from_records(
        DS2_TEST, data_path, valid_leads, out_len, "Test"
    )

    # Create datasets
    train_dataset = ECGDataset(train_data, train_rr, train_labels, patient_id)
    test_dataset = ECGDataset(test_data, test_rr, test_labels, patient_id_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_labels, test_labels


def get_dataloaders(train_data, train_rr, train_labels, patient_id, sample_id,
                    test_data, test_rr, test_labels, patient_id_test, sample_id_test,
                    batch_size: int = BATCH_SIZE) -> tuple:

    train_dataset = ECGDataset(train_data, train_rr, train_labels, patient_id, sample_id)
    test_dataset = ECGDataset(test_data, test_rr, test_labels, patient_id_test, sample_id_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_dataloaders_from_indices(
    data: np.ndarray,
    rr: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    sample_ids: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = BATCH_SIZE,
    dataset_cls=None,
) -> tuple:
    """
    intra-patient split 인덱스로 train/valid/test DataLoader 생성.
    dataset_cls: None이면 ECGDataset 사용
    """
    if dataset_cls is None:
        dataset_cls = ECGDataset

    full_dataset = dataset_cls(data, rr, labels, patient_ids, sample_ids)

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(Subset(full_dataset, valid_idx), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(Subset(full_dataset, test_idx),  batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


# =============================================================================
# MACN Dataset (3-channel input: ECG + RR broadcast)
# =============================================================================

class MACNDataset(Dataset):
    """
    MACN용 Dataset.
    ecg_raw: (N, 128) — raw ECG segments
    rr_3dim: (N, 3)   — [pre_rr, pre_rr_ratio, near_pre_rr_ratio]

    __getitem__ returns:
        x3         (3, 128) — ECG + broadcast RR channels
        rr_3dim    (3,)     — RR features (for model's rr_encoder)
        label      (,)
        patient_id (,)
        idx        (,)
    """

    def __init__(self, ecg_raw, rr_3dim, labels, patient_ids, sample_ids):
        # ecg_raw: (N, 128), rr_3dim: (N, 3)
        self.ecg_raw = torch.FloatTensor(ecg_raw)       # (N, 128)
        self.rr_3dim = torch.FloatTensor(rr_3dim)       # (N, 3)
        self.labels = torch.LongTensor(labels)
        self.patient_ids = torch.LongTensor(patient_ids)
        self.sample_ids = sample_ids  # keep as numpy for Subset compatibility

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ecg_seg = self.ecg_raw[idx].numpy()              # (128,)
        pre_rr_ratio    = float(self.rr_3dim[idx, 1])
        near_pre_rr_ratio = float(self.rr_3dim[idx, 2])

        x3 = torch.FloatTensor(_build_x3(ecg_seg, pre_rr_ratio, near_pre_rr_ratio))  # (3, 128)

        return (x3,
                self.rr_3dim[idx],
                self.labels[idx],
                self.patient_ids[idx],
                idx)




