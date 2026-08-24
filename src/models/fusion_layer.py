import torch
import torch.nn as nn

class TriVectorCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=4096, num_heads=8, dropout=0.1):
        super(TriVectorCrossAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        
        # Multi-Headed Cross-Attention blocks
        self.text_video_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.text_eeg_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Linear layer fusion layer
        self.fusion_projection = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid() # Outputs the bounded Incongruity Index probability score
        )
        
    def forward(self, V_text, V_video, V_eeg):
        """
        Executes parallel cross-modal matching across the tri-vector stack.
        V_text: [Batch_Size, Seq_Len, 4096] -> Acts as the Query (Q)
        V_video: [Batch_Size, Frame_Len, 4096] -> Acts as Key/Value (K, V)
        V_eeg: [Batch_Size, Spectral_Len, 4096] -> Acts as Key/Value (K, V)
        """
        # Cross-Attention Channel 1: Text aligning to Video landmark tensors
        attn_video_out, _ = self.text_video_attention(query=V_text, key=V_video, value=V_video)
        
        # Cross-Attention Channel 2: Text aligning to Quantitative EEG spectral profiles
        attn_eeg_out, _ = self.text_eeg_attention(query=V_text, key=V_eeg, value=V_eeg)
        
        # Pool across temporal/sequential dimensions (Mean Pooling)
        v_text_pooled = torch.mean(V_text, dim=1)
        v_video_pooled = torch.mean(attn_video_out, dim=1)
        v_eeg_pooled = torch.mean(attn_eeg_out, dim=1)
        
        # Concatenate into the integrated tri-vector matrix space
        unified_matrix = torch.cat([v_text_pooled, v_video_pooled, v_eeg_pooled], dim=-1)
        
        # Compute final cross-cohort Incongruity Index batch probability
        incongruity_index_batch = self.fusion_projection(unified_matrix)
        return incongruity_index_batch
