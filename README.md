# NN-Opt

A systematic empirical study of weight initialization strategies, optimization algorithms, and learning rate scheduling techniques in Deep Learning.

## Overview
**NN-Opt** is a research-oriented project designed to evaluate how different components of the training pipeline — from the starting weight distribution to the gradient update logic — influence model convergence, stability, and generalization.

## Core Research Areas

### 1. Weight Initialization & Symmetry Breaking
One of the primary focuses of this project is the impact of initial weight states on gradient flow. We compared "naive" approaches with standard practices to demonstrate the necessity of symmetry breaking.

**Key Findings:**
* **Zero Initialization:** Resulted in a plateau early on. With all weights set to 0, neurons in the same layer perform the same calculation, leading to a lack of diversity in learning.
    * *Accuracy (Train/Val):* 10.88% / 15.24%
* **Constant Initialization (0.1):** Similar to zero-init, this failed to break symmetry effectively, resulting in poor performance.
    * *Accuracy (Train/Val):* 10.51% / 12.36%

This highlights that without proper initialization (like Kaiming or Xavier), even the most advanced optimizers cannot recover the training process.

### 2. Optimizer Benchmark
We conducted a comprehensive comparison of 8 optimization algorithms to analyze their convergence trajectories:
* **Stochastic Gradient Descent (SGD):** Baseline.
* **Momentum-based:** SGD with Momentum, SGD with Nesterov Momentum.
* **Adaptive Methods:** Adagrad, RMSProp.
* **Hybrid/State-of-the-art:** Adam, NAdam, and **AdamW** (decoupled weight decay).

**Analysis:**
Adaptive methods (Adam/AdamW) show significantly faster initial convergence, while Nesterov Momentum often provides smoother loss transitions on specific architectures.

![Uploading Снимок экрана 2026-04-14 в 19.35.22.png…]()


### 3. Learning Rate Scheduling Dynamics
The project evaluates how various scheduling strategies manage the "exploration vs. exploitation" trade-off:
* **LambdaLR:** High flexibility for custom decay functions.
* **StepLR:** Classic discrete reduction.
* **CosineAnnealingWarmRestarts:** Implements periodic "warm restarts" to escape local minima and refine the solution.

**Observations:**
* `CosineAnnealingWarmRestarts` demonstrated superior stability in the later stages of training.
* `LambdaLR` allowed for faster convergence when tuned to the specific loss landscape of the dataset.
<img width="1154" height="472" alt="Снимок экрана 2026-04-14 в 19 20 20" src="https://github.com/user-attachments/assets/702e183a-dd91-41fc-a8e9-79a98d9a9b29" />


## Experimental Setup
* **Framework:** PyTorch
* **Hardware:** Evaluated using NVIDIA T4 GPU (via Google Colab).
* **Epochs:** 30 (for baseline comparisons).
* **Metrics:** Cross-Entropy Loss, Accuracy, Convergence Speed (s).

## Summary of Results
1.  **Initialization is Critical:** Poor initialization (Zero/Constant) makes training practically impossible regardless of the optimizer.
2.  **Optimizer Choice:** AdamW remains the most robust choice for rapid prototyping, though SGD+Nesterov is a strong contender for final fine-tuning.
3.  **Scheduling Matters:** Using a scheduler is mandatory for reaching high accuracy; static learning rates often lead to sub-optimal convergence.

---
*Created as a research portfolio project. Source code and training logs are available in the repository.*
