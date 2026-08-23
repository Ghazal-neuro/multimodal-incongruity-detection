# multimodal-incongruity-detection
A Tri-Vector Multimodal Deep Learning Framework (LLMs + Computer Vision + qEEG) designed to detect subclinical psychopathy and neuro-metabolic cognitive strain."
# Multimodal Incongruity Detection Framework

A Tri-Vector Multimodal Deep Learning Framework (LLMs + Computer Vision + qEEG) designed to detect subclinical psychopathy and neuro-metabolic cognitive strain.

---

## 1. Project Overview

Traditional psychiatric and neuro-metabolic assessments rely heavily on human-administered checklists and subjective evaluations, which are highly vulnerable to either conscious deception (e.g., in subclinical psychopathy) or diagnostic overlay from systemic physiological fatigue (e.g., in Type 1 Diabetes). 

This framework introduces a novel multimodal computational approach designed to map complex subclinical profiles by quantifying **cross-modal incongruity**. By pairing a Large Language Model (LLM) for deep semantic text analysis with a Spatial-Temporal Computer Vision network for facial micro-expression tracking and a Quantitative EEG (qEEG) pipeline for neural metrics, our system maps the systemic discrepancies between an individual's explicit communication and their underlying biological realities.

### Core Architecture Flow
* **Phase A (The "What"):** Audio → Whisper ASR → LLM Context Embeddings → Vector 1 (\(V_{text}\))
* **Phase B (The "How"):** Video → FaceMesh → Spatial-Temporal ViT → Vector 2 (\(V_{video}\))
* **Phase C (The "Internal State"):** Brain Signals → qEEG Processing → Spectral Tensors → Vector 3 (\(V_{eeg}\))
* **Phase D (The Fusion Layer):** Cross-Attention Module → Mathematical Matrix Optimization → **Incongruity Index**

---

## 2. Mathematical Formulation

The core computational contribution of this architecture is a multi-headed cross-attention network that ingests all three normalized vectors to measure their alignment and output the final index score:

\[\text{Incongruity Index} = f(V_{\text{text}}, V_{\text{video}}, V_{\text{eeg}})\]

* **Psychopathic Masking Flag:** Triggered when \(V_{\text{text}}\) signals a high-intensity emotional narrative but \(V_{\text{video}}\) outputs an unmodulated, flat affective baseline alongside an unreactive, detached \(V_{\text{eeg}}\) profile.
* **Neuro-Metabolic Calibration:** Triggered when \(V_{\text{text}}\) displays syntax degradation and cognitive memory load stalls, paired with distinct computer vision fatigue tells and matching, reactive shifts in the EEG Theta/Beta spectrum.

---

## 3. Project Structure & Repository Layout

```text
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py       # Custom Tri-Vector PyTorch Dataset and DataLoaders
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_encoder.py     # Phase A: LLM embedding extractor
│   │   ├── vision_encoder.py   # Phase B: MediaPipe and FACS Action Unit tracker
│   │   ├── eeg_processor.py    # Phase C: MNE-Python signal processing pipeline
│   │   └── fusion_layer.py     # Phase D: PyTorch Joint Cross-Attention layer
│   └── train.py                # Core training loop with False Positive optimization
```

---

## 4. Hardware & Software Requirements

### Hardware Requirements
* **GPU:** Minimum 1x NVIDIA A100 or H100 Tensor Core GPU (80GB VRAM) for concurrent LLM inference and vision transformer operations.
* **RAM:** 128 GB System Memory.
* **Storage:** 2 TB NVMe Solid State Drive (SSD) for continuous high-definition video frame processing.

### Software Dependencies
* **Core Stack:** Python 3.10+, PyTorch (v2.0+) with CUDA acceleration.
* **NLP Modules:** Hugging Face Transformers, spaCy, OpenAI Whisper API.
* **Computer Vision:** MediaPipe FaceMesh, OpenCV, PyTorchVideo.
* **Biosignal Analysis:** MNE-Python.

---

## 5. Targeted Evaluation Metrics

* **Target F1-Score:** $\ge$ 0.82 (optimizing harmonic precision and recall).
* **False Positive Rate (FPR):** Controlled strict limit $\le$ 5% to eliminate misclassifying metabolic brain fog as psychopathic masking.
* **Target AUROC:** $\ge$ 0.88 for clear, robust cohort separation.
* **CV Temporal Sensitivity:** 33 milliseconds (equivalent to 1 frame at 30fps) to catch immediate facial muscle drops.

---

## 6. Supported Datasets

1. **The Real-Life Deception Detection (RLDD) Dataset (University of Michigan):** Multimodal legal and trial video profiles.
2. **The Miami University Deception Database (MU3D) (Miami University):** High-resolution baseline facial action metrics.
3. **In-House Clinical Neuro-Metabolic Database:** Anonymized resting-state functional connectivity, working memory performance, and qEEG records from Type 1 Diabetes (T1D) patient networks.
