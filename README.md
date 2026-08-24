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

Incongruity Index"_(batch) = f ( V_(text) ,V_(video) ,V_(eeg) )

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

## 4.Hyperscale Cloud Infrastructure and AI Deployment Architecture

The production-grade execution graph of this tri-vector cross-attention network is engineered to bypass localized compute bottlenecks by utilizing a fully decoupled, cloud-native distributed architecture. The processing pipeline can be deployed natively across enterprise cloud nodes (AWS, GCP, or Azure) to manage high-throughput, multi-rate ingestion arrays.

### Hardware Infrastructure & Compute Allocation
*   **GPU Acceleration Tier:** Distributed clusters leveraging minimum 1x NVIDIA A100 or H100 Tensor Core GPU (80GB VRAM) to support multi-user batch transformer inference.
*   **Host Allocation:** 128 GB System RAM minimum configuration to prevent buffer overflows during large, parallel tensor data processing loops.
*   **High-Throughput IO:** 2 TB NVMe Solid State Drive (SSD) via PCIe Gen 4 to prevent pipeline stalling during simultaneous frame extraction loops.

### Serving Stack & Containerized Microservices
*   **Container Isolation via Docker:** Each standalone module—`text_encoder.py` (Phase A), `vision_encoder.py` (Phase B), and `eeg_processor.py` (Phase C)—is isolated into localized Docker microservices to ensure complete cross-environment reliability and error isolation.
*   **Orchestration via Kubernetes:** Production pods are orchestrated using centralized Kubernetes configurations utilizing Horizontal Pod Autoscaling (HPA) targets to scale compute assets dynamically based on active clinical intake volume.
*   **High-Performance Serving via NVIDIA Triton:** Model inference and execution grids are served through the **NVIDIA Triton Inference Server**. Triton manages concurrent execution paths and enforces Dynamic Batching across multi-GPU nodes to compute the batch *Incongruity Index* with near-zero latency.
*   **Serverless Ingestion Layer:** The entire PyTorch deep learning backend is exposed via a high-performance, lightweight **gRPC and RESTful API**. Local research stations and clinical facilities can securely stream raw session assets (e.g., standard digital `.edf` or BrainVision files) over an encrypted web hook to receive instantaneous score calculations without requiring local high-performance hardware.
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
