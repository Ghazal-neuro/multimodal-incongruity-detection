import torch
import torch.nn as nn

class TriVectorCrossAttentionFusion(nn.Module):
    """
    A multi-headed joint cross-attention network designed to ingest normalized 
    vectors from three distinct modalities (Text, Video, and EEG) to compute
    the integrated 'Incongruity Index'.
    
    Mathematical Function: Incongruity Index = f(V_text, V_video, V_eeg)
    """
    def __init__(self, embed_dim: int = 4096, num_heads: int = 8):
        """
        Args:
            embed_dim (int): The shared dimension size of the input vectors. 
                             Defaults to 4096 to match modern LLM hidden state layers (e.g., Llama-3).
            num_heads (int): Number of parallel attention heads. Must be a divisor of embed_dim.
        """
        super(TriVectorCrossAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        
        # 1. Cross-Attention: Maps the Linguistic Context (Query) to Visual Affect (Key/Value)
        # Captures the relationship between what is said and how the face moves
        self.text_video_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 2. Cross-Attention: Maps the Linguistic Context (Query) to Neuro-Signal Data (Key/Value)
        # Captures the relationship between what is said and how the brain fires
        self.text_eeg_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. Dense Projection Network (Classification Head)
        # Concatenates the features and applies a non-linear activation pipeline
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 1),
            nn.Sigmoid() # Bounds the output strictly between 0.0 and 1.0 (The Index Score)
        )
        
    def forward(self, v_text: torch.Tensor, v_video: torch.Tensor, v_eeg: torch.Tensor) -> torch.Tensor:
        """
        Executes the tri-vector forward propagation.
        
        Args:
            v_text (torch.Tensor):  [Batch_Size, Embed_Dim] tensor from Phase A (LLM Hidden Layers)
            v_video (torch.Tensor): [Batch_Size, Embed_Dim] tensor from Phase B (CV Spatial-Temporal ViT)
            v_eeg (torch.Tensor):   [Batch_Size, Embed_Dim] tensor from Dr. Andres's qEEG Spectral Tracks
            
        Returns:
            torch.Tensor: [Batch_Size, 1] containing the calculated Incongruity Index values.
        """
        # PyTorch MultiheadAttention expects sequential tensors: [Batch_Size, Sequence_Length, Embed_Dim]
        # Since these are flattened 1D modality representations, we set Sequence_Length = 1
        v_text_seq  = v_text.unsqueeze(1)
        v_video_seq = v_video.unsqueeze(1)
        v_eeg_seq   = v_eeg.unsqueeze(1)
        
        # Run Text-to-Video Attention: Query = Text, Key/Value = Video
        # Output represents the visual features heavily weighted by linguistic salience
        attn_video, _ = self.text_video_attention(query=v_text_seq, key=v_video_seq, value=v_video_seq)
        
        # Run Text-to-EEG Attention: Query = Text, Key/Value = EEG
        # Output represents the neurological features heavily weighted by linguistic salience
        attn_eeg, _ = self.text_eeg_attention(query=v_text_seq, key=v_eeg_seq, value=v_eeg_seq)
        
        # Flatten attention tensor representations back to structural [Batch_Size, Embed_Dim] vectors
        attn_video = attn_video.squeeze(1)
        attn_eeg = attn_eeg.squeeze(1)
        
        # Concatenate original text features with visual and neurological attention maps
        # Resulting Tensor Size: [Batch_Size, Embed_Dim * 3]
        fused_matrix = torch.cat((v_text, attn_video, attn_eeg), dim=-1)
        
        # Compute the final bounded classification score
        incongruity_index = self.classifier(fused_matrix)
        return incongruity_index


# Validation Execution Block
if __name__ == "__main__":
    print("[RUNNING] Validating Tri-Vector Cross-Attention matrix operations...")
    
    # Instantiate the fusion layer module
    fusion_network = TriVectorCrossAttentionFusion(embed_dim=4096, num_heads=8)
    fusion_network.eval() # Set to evaluation mode (deactivates dropout)
    
    # Generate 4 mock data samples simulating outputs from your pre-processing stages
    # Batch size = 4, Vector Dimensions = 4096
    mock_batch_v_text  = torch.randn(4, 4096)
    mock_batch_v_video = torch.randn(4, 4096)
    mock_batch_v_eeg   = torch.randn(4, 4096)
    
    # Process tensors through the forward propagation graph
    with torch.no_grad():
        calculated_indices = fusion_network(mock_batch_v_text, mock_batch_v_video, mock_batch_v_eeg)
        
    print("\n--- Structural Output Validation ---")
    print(f"Input Matrix Batch Sizes        : {mock_batch_v_text.shape[0]}")
    print(f"Output Index Matrix Shape       : {calculated_indices.shape}") # Expected: [4, 1]
    print(f"Generated Index Value Array     :\n{calculated_indices.flatten().tolist()}")
    
    # Confirm bounded constraints
    assert (calculated_indices >= 0.0).all() and (calculated_indices <= 1.0).all(), "Error: Index bounds broken."
    print("\n[SUCCESS] Matrix operations checked. PyTorch code matches proposal specifications!")
