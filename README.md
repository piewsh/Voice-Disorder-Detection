# 🗣️ Voice Disorder Detection & Classification

> **Online Sequential Extreme Learning Machine (OS-ELM) for medical voice pathology detection using MFCC features**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)

---

## 📋 What This Does

A machine learning pipeline for detecting and classifying voice disorders from audio signals. The system extracts MFCC features from voice recordings and uses a custom **PyTorch OS-ELM** (Online Sequential Extreme Learning Machine) implementation for fast, incremental classification — suitable for real-time clinical applications.

---

## 🧠 Core: OS-ELM Implementation (`oselm.py`)

A full PyTorch implementation of the Online Sequential Extreme Learning Machine:

- **Architecture**: Single hidden-layer feedforward network with random input weights (non-gradient)
- **Online Learning**: Supports incremental training — new data updates the model without retraining from scratch using recursive least squares (Woodbury matrix identity)
- **GPU Accelerated**: All matrix operations run on CUDA when available
- **Two-Phase Training**:
  1. **Initialization Phase** — Computes initial output weights via pseudo-inverse: `β = (H^T H)^{-1} H^T T`
  2. **Sequential Phase** — Updates weights incrementally: `P_{k+1}`, `β_{k+1}` via Woodbury formula

---

## 📊 Pipeline

### Preprocessing (`preprocessing.py`)
- Audio signal loading and normalization
- **MFCC** (Mel-Frequency Cepstral Coefficients) feature extraction
- Feature scaling and preparation

### Detection (`detection.ipynb`)
- Binary classification: healthy vs. pathological voice
- Dataset exploration and feature visualization
- **SMOTE** (Synthetic Minority Oversampling) for class imbalance handling

### Classification (`classification.ipynb`)
- Multi-class voice disorder classification
- Model training, validation, and evaluation
- Performance metrics: accuracy, confusion matrix, classification report

---

## 📁 Project Structure

```
Voice-Disorder-Detection/
├── oselm.py              # PyTorch OS-ELM implementation (92 lines)
├── preprocessing.py       # MFCC feature extraction pipeline
├── detection.ipynb        # Binary detection notebook
├── classification.ipynb   # Multi-class classification notebook
└── README.md
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | OS-ELM (custom PyTorch implementation) |
| Features | MFCC (Librosa) |
| Balancing | SMOTE (Imbalanced-Learn) |
| Framework | PyTorch |
| Analysis | Jupyter, Scikit-Learn, NumPy |

## 👤 Author

**Piyush Ranjan Singh** — [GitHub](https://github.com/piewsh) • [Email](mailto:rajputpiyush2009@gmail.com)
