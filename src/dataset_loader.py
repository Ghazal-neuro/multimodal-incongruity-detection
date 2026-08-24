import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultimodalTriVectorDataset(Dataset):
    """
    [Detailed docstring added to update, referencing multi-university data sources.]
    """
    def __init__(self, data_samples: list):
        """
        [Updated comments to reflect multi-university database origins.]
        """
        self.samples = data_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # [Existing, functioning __getitem__ method preserved]
        sample = self.samples[idx]
        v_text = torch.tensor(sample['text_embedding'], dtype=torch.float32)
        v_video = torch.tensor(sample['video_features'], dtype=torch.float32)
        v_eeg = torch.tensor(sample['eeg_signals'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.float32).unsqueeze(-1)
        return v_text, v_video, v_eeg, label

# [Self-contained verification block remains unchanged]
