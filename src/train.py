import torch
import torch.nn as nn
import torch.optim as optim
from models.fusion_layer import TriVectorCrossAttentionFusion

class AsymmetricFPRLoss(nn.Module):
    def __init__(self, penalty_weight=5.0, eps=1e-7):
        super(AsymmetricFPRLoss, self).__init__()
        self.penalty_weight = penalty_weight
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Enforces a strict 5x gradient penalty exclusively on False Positive configurations.
        y_true: 0 = Type 1 Diabetes (Metabolic Fatigue), 1 = Callous-Unemotional Traits (Behavioral Mask)
        """
        # Clamp bounds to avoid log(0) NaN instabilities
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        
        # Standard Binary Cross Entropy components
        true_positive_loss = y_true * torch.log(y_pred)
        false_positive_loss = (1.0 - y_true) * torch.log(1.0 - y_pred)
        
        # Apply directional 5.0 penalty multiplier to the false positive tracking layer
        loss_tensor = -(true_positive_loss + self.penalty_weight * false_positive_loss)
        return torch.mean(loss_tensor)

def train_hyperscale_batch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (V_text, V_video, V_eeg, labels) in enumerate(dataloader):
        V_text, V_video, V_eeg = V_text.to(device), V_video.to(device), V_eeg.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Compute multi-modal batch index probabilities
        incongruity_scores = model(V_text, V_video, V_eeg)
        
        # Run custom optimization to squeeze False Positives strictly under the 5% ceiling
        loss = criterion(incongruity_scores, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)
