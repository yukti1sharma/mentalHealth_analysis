# 🧠 Multimodal Fusion for Depression Severity Estimation

> Multimodal fusion of facial dynamics and clinical interview transcripts for automated PHQ-8 depression severity prediction.

**Anushka Kathil · Dhruvi Sharma · Apoorva Bajpai**  
*Supervisor: Dr. Afreen Khursheed | ECE Department, IIIT Bhopal*

---

## The Problem

Depression affects **280M+ people worldwide** (WHO), yet clinical assessment remains heavily subjective.

| Challenge | Detail |
|-----------|--------|
| **Subjective Assessment** | PHQ-8 relies on self-report — vulnerable to anosognosia and social stigma |
| **Clinician Variability** | Assessments differ by training, experience, and cultural background |
| **Data Scarcity** | E-DAIC has only 275 participants — too few for large end-to-end deep learning |
| **Class Imbalance** | 'None' = 44.4% vs 'Severe' = 3.6%, making fair classification inherently hard |

---

## Project Objectives

1. Design a **temporal attention-based BiLSTM** for facial Action Unit dynamics from clinical interview videos
2. Integrate **MentalRoBERTa** for clinically meaningful semantic representations from interview transcripts
3. Develop a **multimodal SVM fusion** strategy combining facial and text embeddings
4. Compare against deep cross-attention fusion and unimodal baselines using **5-fold stratified CV**
5. Conduct ablation studies with interpretable analysis via attention weights & t-SNE

---

## Dataset: E-DAIC

| | |
|---|---|
| **275** participants | Semi-structured clinical interviews |
| **5** modalities | Video, Audio, Action Units, Transcripts, PHQ-8 scores |
| **17** Action Units | Extracted by OpenFace 2.1.0 at ~30 fps |

Severity labels follow the standard PHQ-8 banding:

| Class | Label | PHQ-8 Range |
|-------|-------|-------------|
| 0 | None | 0 – 4 |
| 1 | Mild | 5 – 9 |
| 2 | Moderate | 10 – 14 |
| 3 | Moderate-Severe | 15 – 19 |
| 4 | Severe | 20 – 27 |

---

## System Architecture

```
BRANCH A — Facial
  OpenFace AU Features (17 AUs @ ~30fps)
    → Confidence Filtering
    → Temporal Downsampling
    → Fixed Window (3000 timesteps)
    → BiLSTM Layer 1: 128 units/dir → 256-d + LayerNorm
    → BiLSTM Layer 2:  64 units/dir → 128-d + LayerNorm
    → Temporal Self-Attention: e_t = wᵀh_t + b → softmax → weighted sum
    → z_face (128-d)

BRANCH B — Text
  Clinical Transcript
    → Filter Participant Utterances
    → Concatenate Utterances
    → RoBERTa Tokenisation (512 tokens)
    → MentalRoBERTa Encoding  [mental/mental-roberta-base]
    → Mean Pooling over all token embeddings
    → z_text (768-d)

FUSION
  Concatenate [z_face ‖ z_text] → x_fused (896-d)
    → StandardScaler
    → PCA (100 components)
    → SVM-RBF (C=10, class_weight='balanced')
    → PHQ-8 Severity Class: 0 (None) → 4 (Severe)
```

---

## Model Details

### Branch A — BiLSTM + Temporal Attention

The BiLSTM processes sequences of 3000 × 17 Action Unit tensors per participant.

- **Why Bidirectional?** The forward LSTM captures how expressions evolve over time; the backward pass captures anticipatory patterns. Together they provide full-context temporal representation.
- **Why Temporal Attention?** Rather than relying on the final LSTM state (which over-emphasises the interview end), attention learns a weighted sum across all 3000 timesteps, pinpointing diagnostically relevant segments.

### Branch B — MentalRoBERTa

| Property | Detail |
|----------|--------|
| Model | `mental/mental-roberta-base` |
| Architecture | RoBERTa-base — 12 layers, 12 attention heads |
| Parameters | ~125 million |
| Pre-training | Reddit mental health communities (r/depression, r/anxiety) |
| Output | 768-d mean-pooled embedding |

- **Why no fine-tuning?** A model fine-tuned on Kaggle mental health data (34k samples) achieved F1=0.953 on that set but performed *worse* on clinical transcripts — social media language ≠ clinical interview language.
- **Why mean pooling?** Mean pooling over all 512 token embeddings outperforms `[CLS]`-only for long conversational clinical text.

### Fusion — SVM-RBF

- **Why SVM over deep fusion?** The cross-attention transformer collapsed — predicting mean PHQ-8 (~7) for all participants (F1-binary=0.000). With only 275 samples, there is insufficient data to train stable cross-modal attention. SVM's structural risk minimisation provides robust generalisation on high-dimensional small-sample problems.
- Feature:sample ratio of 896:275 necessitates PCA dimensionality reduction before classification.

---

## Results

### Ablation Study

| Configuration | MAE ↓ | F1-binary ↑ | F1-macro ↑ |
|---------------|-------|-------------|------------|
| Face only — BiLSTM + Attention | 5.09 | 0.000 | 0.147 |
| Deep Fusion — Cross-Attention | 5.03 | 0.000 | 0.054 |
| **SVM Fusion (Proposed)** | **5.58** | **0.292** | **0.240** |

> The facial BiLSTM achieves MAE=5.09, surpassing the **AVEC 2019 official baseline (MAE=6.50)**.  
> F1-binary (depressed vs. non-depressed at PHQ ≥ 10) is the primary clinical metric.

### Classifier Comparison (896-d Fused Embeddings, 5-Fold CV)

| Model | MAE ↓ | F1-macro ↑ | F1-binary ↑ |
|-------|-------|-----------|------------|
| Random Forest | 5.673 | 0.139 | — |
| Gradient Boosting | 5.749 | 0.158 | — |
| **SVM-RBF (best)** | **5.582** | **0.240** | **0.292** |

SVM-RBF consistently outperforms Random Forest and Gradient Boosting across all metrics on the same 896-dimensional fused embeddings.

> The 'None' class is correctly identified in **69% of cases** (84/122) — practically useful for clinical pre-screening.

---

## Dataset Access

This project uses the **E-DAIC (Extended Distress Analysis Interview Corpus)**, which requires a data use agreement from the USC Institute for Creative Technologies. The dataset is **not included** in this repository.

- Homepage: [dcapswoz.ict.usc.edu](http://dcapswoz.ict.usc.edu)

---
