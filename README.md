# Modelling of Magnetic Hysteresis Loops under Distorted Excitation by using Deep Neural Networks and an Attention Mechanism

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-success.svg)]()

Official open-access code and benchmark repository for the paper:

> **Modelling of Magnetic Hysteresis Loops under Distorted Excitation by using Deep Neural Networks and an Attention Mechanism**  
> *Davide Milillo, Luigi Sabino, Michele Quercio, and Francesco Riganti Fulginei*  
> Submitted to **Applied Soft Computing** (Elsevier), Manuscript ID: `ASOC-D-26-08391`.

---

## 📖 Overview

Predicting ferromagnetic hysteresis under complex, non-sinusoidal waveforms with dense harmonic distortion is critical for magnetic component design, power electronics, and transient circuit simulation. While classical phenomenological operators such as the **Preisach model** are accurate, their numerical evaluation is computationally intensive and non-trivial to vectorize within differential equation solvers.

This repository provides an open, fully reproducible benchmark of **deep neural sequence models** and **attention mechanisms** for magnetic hysteresis prediction under strict, realistic evaluation conditions:

1. **Closed-Loop Autoregressive Rollout**: Unlike many studies in the literature that evaluate sequence models under *teacher forcing* (open-loop inference with true ground-truth history), all models here are evaluated in a true closed-loop autoregressive regime. After bootstrapping with an initial window ($T_w = 70$ steps), the models must recursively feed their own predicted magnetization back into the input sequence.
2. **Causal vs. Anticipated Information Budgets**: We rigorously distinguish between causal single-step autoregressive models (no future knowledge of excitation) and multi-horizon transformers with known future input covariates.
3. **Physical & Topological Metrics**: Beyond standard regression metrics ($R^2$, RMSE, MAE), models are evaluated on cycle-dissipated energy (**Area Error %**), **Coercive Field Error** ($H_c$), and **Remanent Magnetization Error** ($M_{rem}$ at zero-field crossings).

---

## 🧠 Evaluated Architectures

| Model Class | Architecture Description | Information Budget | Prediction Horizon |
| :--- | :--- | :--- | :--- |
| **LSTM** | Standard deep recurrent baseline (2 layers: 300 and 100 hidden units with dropout) | Causal past only | 1 step (autoregressive) |
| **SelfAttn-LSTM** | Multi-head self-attention on input features (2 heads, 32 keys) + LSTM sequence reducer (32 units) | Causal past only | 1 step (autoregressive) |
| **Enc-SelfAttn** | LSTM Encoder (64 units) + Scaled Dot-Product Attention over latent encoder states (1 head, 64 keys) + LSTM Decoder (32 units) | Causal past only | 1 step (autoregressive) |
| **TFT** | Temporal Fusion Transformer with Variable Selection Networks, Gated Residual Networks, and Multi-Head Interpretable Attention | Future covariates known ($u$, $\dot{u}$) | 25 steps (block recursive) |
| **TFT (No Future)** | Ablation variant of the TFT strictly blinded to future excitation | Causal past only | 25 steps (block recursive) |

---

## 📁 Repository Structure

```text
magnetic-hysteresis-attention-modeling/
│
├── README.md                      # Complete documentation & reproduction guide
├── LICENSE                        # Open-source MIT license
├── .gitignore                     # Git ignore rules for MATLAB & OS files
│
├── data/                          # Data generation & physical simulation
│   ├── Preisach.m                 # Standalone Preisach operator (10,201 switching elements)
│   ├── hysteresis_training.m      # Deterministic generator for 40 train curves & 7 test curves
│   └── train_curves.m             # Interactive waveform explorer & Preisach simulator
│
├── models/                        # Network architectures & training routines
│   ├── layers/                    # Custom MATLAB deep learning layers
│   │   ├── gluNetworkLayer.m      # Gated Linear Unit (GLU)
│   │   ├── grnNetworkLayer.m      # Gated Residual Network (GRN)
│   │   ├── interpretableSelfAttentionNetworkLayer.m
│   │   ├── separateChannels.m     # Channel separation utility for dlnetwork
│   │   └── variableSelectionNetworkLayer.m
│   ├── createTFTNetwork.m         # Architecture constructor for Temporal Fusion Transformer
│   ├── train_LSTM.m               # Training script for LSTM baseline
│   ├── train_SelfAttn_LSTM.m      # Training script for SelfAttn-LSTM model
│   ├── train_Enc_SelfAttn.m       # Training script for Enc-SelfAttn model
│   ├── train_TFT.m                # Training script for TFT model
│   └── train_TFT_no_future.m      # Training script for TFT (No Future) causal ablation
│
├── evaluation/                    # Benchmarking & paper figure reproduction
│   ├── compare_all_models.m       # Aggregate benchmark: tables and summary metrics
│   ├── compare_detail_models.m    # Generates 20 publication-ready M-H comparison plots
│   ├── generate_paper_tables.m    # Generates formal LaTeX tables (booktabs style)
│   └── interpretability_analysis.m# Occlusion sensitivity & reversal point alignment
│
└── results_comparison/            # Pre-computed simulation outputs & checkpoints
    ├── results_LSTM_*.mat         # Checkpoint and rollout trajectories for LSTM
    ├── results_SelfAttention_*.mat# Checkpoint and rollout trajectories for SelfAttn-LSTM
    ├── results_Bahdanau_*.mat     # Checkpoint and rollout trajectories for Enc-SelfAttn
    ├── results_TFT_*.mat          # Checkpoint and rollout trajectories for TFT
    ├── results_TFT_NO_FUTURE_*.mat# Checkpoint and rollout trajectories for TFT (No Future)
    ├── figures/                   # Comparative figures & attention maps
    └── figures_detailed/          # High-resolution individual M-H hysteresis cycle plots
```

---

## ⚡ Quickstart: 1-Click Reproduction of Paper Results

You do **not** need to re-train the neural networks to inspect the outputs. All pre-trained trajectories and closed-loop rollouts across all 7 test waveforms are stored in `results_comparison/`.

### 1. Reproduce Benchmark Tables and Metrics
Open MATLAB and run:
```matlab
cd('evaluation');
compare_all_models;
```
This prints the full metric breakdown (RMSE, MAE, $R^2$, Area Error %, Coercive Field Error, and Remanent Magnetization Error) for both **In-Domain** and **Extended (Out-of-Domain)** test sets.

### 2. Export Publication LaTeX Tables
```matlab
cd('evaluation');
generate_paper_tables;
```
This generates `results_comparison/LaTeX_Tables.txt`, containing LaTeX tables in `booktabs` format matching Tables 2 through 10 in the manuscript.

### 3. Generate Detailed M-H Curve Figures
```matlab
cd('evaluation');
compare_detail_models;
```
This creates high-resolution (300 DPI) comparison plots of model trajectories versus the Preisach ground truth in `results_comparison/figures_detailed/`.

---

## 🔬 Dataset Generation & Training Protocol

### Physical Preisach Simulator
* Discretization grid: $N_{\text{grid}} = 101 \times 101 = 10,201$ elementary switching operators over $(\alpha, \beta) \in [-1, 1]$.
* Weight distribution: Bivariate Gaussian distribution $\mu(\alpha, \beta) = \exp(-(\alpha^2 + \beta^2))$ for $\alpha \ge \beta$.
* Time resolution: Fundamental frequency $f_1 = 1.43$ Hz ($T = 0.7$ s), total duration $t_{\text{final}} = 4T = 2.8$ s, sampled at 2,000 discrete points ($\Delta t = 1.4$ ms).

### Dataset Partitioning
* **Training Set (40 curves)**:
  * 10 $\times$ Multi-Sine (Type 1): Saturation-driven fundamental plus 3rd–7th harmonic distortion.
  * 10 $\times$ Concentric Loops (Type 5): Nested minor loops tracing intermediate reversal states.
  * 10 $\times$ FORC (Type 6): First-Order Reversal Curves (5 top-driven, 5 bottom-driven).
  * 10 $\times$ Dense Minor Loops (Type 7): High-frequency oscillatory returns.
* **Test Set (7 curves)**:
  * 4 In-Domain: Multi-Sine, Concentric Loops, FORC, Dense Minor Loops.
  * 3 Out-of-Domain: Damped Sine (exponential envelope decay), Stepwise (discontinuous levels), Triangular (constant $|du/dt|$).

### Training Neural Models from Scratch
To train any of the models from scratch, simply run the corresponding script in `models/`:
```matlab
cd('models');
train_LSTM;              % Standard LSTM baseline
train_SelfAttn_LSTM;     % SelfAttn-LSTM hybrid
train_Enc_SelfAttn;      % Enc-SelfAttn architecture
train_TFT;               % Temporal Fusion Transformer (future known)
train_TFT_no_future;     % TFT causal ablation (future unknown)
```
Trained network rollouts and evaluation summaries are automatically timestamped and saved into `results_comparison/`.

---

## 📊 Summary of Benchmark Performance

### In-Domain Performance (Average across Seen Waveforms)
| Model | RMSE | MAE | $R^2$ | Area Err (%) | Coercive Err | Remanent Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | 0.1144 | 0.0896 | 0.9730 | 22.21% | 0.0350 | 0.1100 |
| **SelfAttn-LSTM** | 0.0970 | 0.0779 | 0.9818 | 13.94% | 0.0437 | 0.1294 |
| **Enc-SelfAttn** | **0.0516** | **0.0419** | **0.9948** | **6.72%** | **0.0225** | **0.0517** |
| **TFT (Anticipated)** | 0.0305 | 0.0225 | 0.9979 | 6.38% | 0.0115 | 0.0286 |
| **TFT (No Future)** | 0.0822 | 0.0626 | 0.9860 | 15.44% | 0.0423 | 0.0473 |

### Out-of-Domain Performance (Average across Unseen Waveforms)
| Model | RMSE | MAE | $R^2$ | Area Err (%) | Coercive Err | Remanent Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | 0.2089 | 0.1778 | 0.7836 | 65.29% | 0.1418 | 0.1685 |
| **SelfAttn-LSTM** | 0.1847 | 0.1567 | 0.8373 | 75.48% | 0.1930 | 0.2660 |
| **Enc-SelfAttn** | **0.1258** | **0.0953** | **0.9385** | **53.07%** | **0.1491** | **0.1525** |
| **TFT (Anticipated)** | 0.0879 | 0.0716 | 0.9631 | 19.70% | 0.0157 | 0.0757 |
| **TFT (No Future)** | 0.2676 | 0.2027 | 0.7648 | 80.48% | 0.2541 | 0.2688 |

*Takeaway*: Under strictly causal information budgets, **Enc-SelfAttn** decisively outperforms both purely recurrent LSTMs and TFT without future covariates, preserving physical trajectory fidelity on nested minor loops and axis crossings.

---

## 🛠️ Software Requirements

* **MATLAB**: R2023b or R2024b (tested and verified on MATLAB R2024b).
* **Required Toolboxes**:
  * Deep Learning Toolbox
  * Signal Processing Toolbox
  * Statistics and Machine Learning Toolbox

---

## 📜 Citation

If you use this codebase or benchmark in your research, please cite our article:

```bibtex
@article{milillo2026hysteresis,
  title     = {Modelling of Magnetic Hysteresis Loops under Distorted Excitation by using Deep Neural Networks and an Attention Mechanism},
  author    = {Milillo, Davide and Sabino, Luigi and Quercio, Michele and Riganti Fulginei, Francesco},
  journal   = {Applied Soft Computing},
  year      = {2026},
  note      = {Under Review, Manuscript ID: ASOC-D-26-08391}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
