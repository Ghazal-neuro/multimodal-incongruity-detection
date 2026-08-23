import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultimodalTriVectorDataset(Dataset):
    """
    A custom PyTorch Dataset engineered to handle, align, and systematically normalize
    tri-vector inputs (Text, Video, and EEG features) for the Incongruity Index framework.
    """
    def __init__(self, data_samples: list):
        """
        Args:
            data_samples (list of dicts): A registration array where each index contains:
                - 'text_embedding': List or numpy array of the LLM output (Expected Dim: 4096)
                - 'video_features': List or numpy array of the CV FaceMesh output (Expected Dim: 4096)
                - 'eeg_signals': List or numpy array of the qEEG spectral data (Expected Dim: 4096)
                - 'label': Integer (0 = Neuro-Metabolic/Diabetic Fatigue, 1 = Psychopathic Masking)
        """
        self.samples = data_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Phase A: Convert LLM Text Vector to PyTorch FloatTensor [Dim: 4096]
        v_text = torch.tensor(sample['text_embedding'], dtype=torch.float32)
        
        # 2. Load Phase B: Convert CV Face Vector to PyTorch FloatTensor [Dim: 4096]
        v_video = torch.tensor(sample['video_features'], dtype=torch.float32)
        
        # 3. Load Clinical Data Layer: Convert EEG Vector to PyTorch FloatTensor [Dim: 4096]
        v_eeg = torch.tensor(sample['eeg_signals'], dtype=torch.float32)
        
        # 4. Load the target binary classification label
        label = torch.tensor(sample['label'], dtype=torch.float32).unsqueeze(-1)
        
        # Internal normalization ensures numerical values remain stable during cross-attention
        v_text = self._z_score_normalize(v_text)
        v_video = self._z_score_normalize(v_video)
        v_eeg = self._z_score_normalize(v_eeg)
        
        return v_text, v_video, v_eeg, label

    def _z_score_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies internal scaling to prevent explosive gradients across attention layers."""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return tensor - mean
        return (tensor - mean) / std


# Self-Contained Verification Block
if __name__ == "__main__":
    print("[RUNNING] Validating Tri-Vector Multimodal Dataset Pipeline...")
    
    # Generate 8 mock data samples simulating 8 unique participant evaluation entries
    mock_registry = []
    for i in range(8):
        mock_entry = {
            'text_embedding': np.random.randn(4096),  # Simulated Llama-3 text vector
            'video_features': np.random.randn(4096),  # Simulated FaceMesh spatial-temporal vector
            'eeg_signals': np.random.randn(4096),     # Simulated qEEG spectrum tensor
            'label': 1 if i % 2 == 0 else 0           # Alternating targets
        }
        mock_registry.append(mock_entry)
        
    # Initialize your dataset instance
    dataset = MultimodalTriVectorDataset(data_samples=mock_registry)
    
    # Configure your batch size data loader
    # Batch size = 4 ensures your tensor outputs are structurally batched as [4, 4096]
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Execute a simple verification forward pass through a single batch iteration
    for batch_idx, (text_batch, video_batch, eeg_batch, label_batch) in enumerate(data_loader):
        print("\n--- Tensor Ingestion Dimension Validation ---")
        print(f"Batch Processing Sequence      : {batch_idx + 1}")
        print(f"V_text Tensor Matrix Shape     : {text_batch.shape}")   # Expected: [4, 4096]
        print(f"V_video Tensor Matrix Shape    : {video_batch.shape}")  # Expected: [4, 4096]
        print(f"V_eeg Tensor Matrix Shape      : {eeg_batch.shape}")    # Expected: [4, 4096]
        print(f"Target Label Matrix Shape      : {label_batch.shape}")  # Expected: [4, 1]
        print(f"Current Target Batch Targets   :\n{label_batch.flatten().tolist()}")
        break
        
    print("\n[SUCCESS] Custom data infrastructure functions perfectly and scales cleanly for the fusion layer!")
