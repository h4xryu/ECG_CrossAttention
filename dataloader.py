
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_PATH, VALID_LEADS, OUT_LEN, BATCH_SIZE, DS1_TRAIN, DS2_TEST
from utils import extract_beats_and_rr_from_records

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




