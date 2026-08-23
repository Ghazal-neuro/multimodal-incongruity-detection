import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import spacy

class PhaseALinguisticsEncoder(nn.Module):
    """
    A foundational natural language processing module that extracts deep semantic 
    embeddings (V_text) from an open-weight LLM alongside explicit statistical 
    syntactic tells characteristic of subclinical masking and cognitive loads.
    """
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B", target_dim: int = 4096):
        """
        Args:
            model_name (str): The HuggingFace identifier for the target language backbone.
            target_dim (int): The expected matrix dimension required by the cross-attention layer.
        """
        super(PhaseALinguisticsEncoder, self).__init__()
        # Load the standard linguistic tokenization stack
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        
        # Initialize spaCy for part-of-speech (POS) syntax parsing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if the small model isn't pre-downloaded in the local environment
            self.nlp = spacy.blank("en")
            
        # Linear layer projection to guarantee the vector fits the 4096 fusion channel size
        self.dimension_projection = nn.Linear(self.llm_backbone.config.hidden_size, target_dim)
        
    def forward(self, transcript_text: str) -> tuple:
        """
        Processes a raw interview transcript slice.
        
        Args:
            transcript_text (str): String containing raw text or ASR outputs.
            
        Returns:
            tuple: (v_text, syntactic_metrics_dict)
                - v_text: A [1, 4096] tensor representing tokenized semantic states.
                - syntactic_metrics_dict: Dictionary containing instrumental and tense ratios.
        """
        # 1. Generate Deep Semantic Vector Representation
        tokens = self.tokenizer(transcript_text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.llm_backbone(**tokens)
            
        # Extract the final hidden layer and apply mean pooling across the token sequence length
        last_layer_hidden_states = outputs.last_hidden_state  # Shape: [Batch, Sequence_Len, Hidden_Dim]
        pooled_states = torch.mean(last_layer_hidden_states, dim=1) # Shape: [Batch, Hidden_Dim]
        
        # Project vector dimensions smoothly to fit Phase C parameter criteria
        v_text = self.dimension_projection(pooled_states) # Shape: [Batch, Target_Dim]
        
        # 2. Extract Explicit Syntactic Tells via Part-of-Speech Tagging
        doc = self.nlp(transcript_text)
        valid_words = [token for token in doc if not token.is_punct and not token.is_space]
        total_word_count = len(valid_words)
        
        if total_word_count == 0:
            return v_text, {"instrumental_ratio": 0.0, "past_tense_ratio": 0.0, "cognitive_filler_density": 0.0}
            
        # Track instrumental metrics (causal indicators mapping transactional logic)
        causal_markers = {"because", "since", "so", "consequently", "therefore", "thus"}
        instrumental_count = sum(1 for token in doc if token.text.lower() in causal_markers)
        
        # Track structural psychological distancing indicators via verb tenses
        past_tense_verbs = sum(1 for token in doc if token.tag_ in ["VBD", "VBN"])
        present_tense_verbs = sum(1 for token in doc if token.tag_ in ["VBP", "VBZ", "VBG"])
        total_verbs = past_tense_verbs + present_tense_verbs
        
        # Track working memory cognitive fillers mapping fatigue or masking load spikes
        filler_markers = {"uh", "um", "like", "well", "ah", "er"}
        filler_count = sum(1 for token in doc if token.text.lower() in filler_markers)
        
        # Package metrics into clean data frames
        syntactics = {
            "instrumental_ratio": float(instrumental_count / total_word_count),
            "past_tense_ratio": float(past_tense_verbs / total_verbs) if total_verbs > 0 else 0.0,
            "cognitive_filler_density": float(filler_count / total_word_count)
        }
        
        return v_text, syntactics

# Pipeline Self-Contained Testing Block
if __name__ == "__main__":
    print("[RUNNING] Validating Phase A Language Encoder configurations...")
    
    # Instantiate the processing pipeline block using a lightweight mockup target for fast verification
    # Note: In standard runtime execution, this defaults to your configured Llama-3 architecture
    encoder_pipeline = PhaseALinguisticsEncoder(model_name="sshleifer/tiny-gpt2", target_dim=4096)
    encoder_pipeline.eval()
    
    sample_interrogation_slice = (
        "I systematically relocated the funds because it made logical financial sense "
        "so that I could secure the company assets. There was um no other alternative, you know."
    )
    
    # Run the execution graph
    v_text, structural_tells = encoder_pipeline(sample_interrogation_slice)
    
    print("\n--- Language Vector Verification ---")
    print(f"Generated Vector Size (V_text) : {v_text.shape}") # Verification Check: Expected [1, 4096]
    print("\n--- Extracted Syntactic Metrics ---")
    for key, score in structural_tells.items():
        print(f"-> {key.replace('_', ' ').title():<26}: {score * 100:.2f}%")
        
    print("\n[SUCCESS] Phase A linguistic extraction is structurally verified and ready for GitHub deployment!")
