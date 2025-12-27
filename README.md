# Food-101 Classification: 2-Stage Training Strategy from Scratch

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Status](https://img.shields.io/badge/Status-Completed-success)

> **Training a ResNet-18 model from scratch on the Food-101 dataset without pre-trained weights, achieving 73.12% Top-1 Accuracy.**

## 📌 Project Overview
The goal of this project was to achieve significant performance on the **Food-101 dataset (101 classes)** by training a Deep Learning model **from scratch**, without relying on pre-trained weights (e.g., ImageNet).

To address common challenges such as **overfitting** and **slow convergence** when training from scratch, I designed a **Custom ResNet Architecture** and a unique **2-Stage Progressive Augmentation Pipeline**.

## 🏗️ Architecture & Methodology

### 1. Custom ResNet Implementation
Instead of using `torchvision.models`, I implemented the **ResNet-18 architecture from scratch** to fully understand the residual learning framework and ensure model lightweighting.
* **Residual Block:** Implemented Skip Connections (`F(x) + x`) to prevent the Vanishing Gradient problem.
* **Structure:** A 4-layer structure based on `BasicBlock` (Filters: 64 → 128 → 256 → 512).

### 2. 2-Stage Training Strategy (Core Feature 💡)
To maximize training efficiency, I divided the training process into two distinct stages with different augmentation intensities.

| Stage | Goal | Augmentation Strategy | Learning Rate |
|:---:|:---|:---|:---:|
| **Stage 1** | **Representation Learning** <br> (Learning general features) | **Strong Augmentation** <br> (ColorJitter, Rotation, Crop) | 0.1 (CosineAnnealing) |
| **Stage 2** | **Fine-tuning** <br> (Refining decision boundaries) | **Weak Augmentation** <br> (No Color Distortion) | 0.0005 (Low LR) |

* **Insight:** In the early stage, strong distortions were applied to force the model to learn general features such as shape and texture. In the later stage, the augmentation was relaxed to fine-tune the model under a distribution similar to the real test data.

## 📊 Experiments & Results

### Performance Comparison
| Model | Training Method | Top-1 Accuracy | Note |
|:---|:---|:---:|:---|
| **Custom ResNet-18** | **2-Stage Strategy (Ours)** | **73.12%** | **Target Achieved 🎯** |
| Custom ResNet-18 | Standard Training | 55.85% | Baseline |
| ResNet-50 | Transfer Learning (ImageNet) | 77.97% | Upper Bound |

* **Analysis:** By applying the 2-Stage strategy, I achieved a **~17% accuracy improvement** over the baseline and significantly narrowed the gap with the pre-trained model.

### Error Analysis (SVM)
To verify the **Feature Extraction Capability** of the trained model, I froze the backbone and attached an SVM classifier to the output features.
* **SVM Accuracy:** 69.84%
* **Failure Cases:** The model struggled to distinguish between classes with similar sauce colors and textures, such as **"Seasoned Chicken (Yangnyeom-chicken)" vs. "Spicy Garlic Fried Chicken (Kkanpunggi)"**.

## 📂 Directory Structure
```bash
Food101-ResNet-Scratch/
├── assets/                  # Images for README
├── src/                     # Source Code
│   ├── model.py             # Custom ResNet Implementation
│   └── transforms.py        # 2-Stage Augmentation Logic
├── experiments/             # Analysis Scripts
│   ├── transfer_learning.py # Baseline Comparison
│   └── svm_analysis.py      # Feature Extraction Analysis
├── train.py                 # Main Training Loop
└── requirements.txt         # Dependencies
```

## 🚀 Usage
1. Install Dependencies
```
pip install -r requirements.txt
```
2. Train Model (2-Stage)
```
python train.py
```
3. Run Analysis (Transfer Learning or SVM)
```
python experiments/transfer_learning.py
# or
python experiments/svm_analysis.py
```

### 💡 Additional Tip for `requirements.txt`

If you haven't created the `requirements.txt` file yet, here is the content for it as well:

```text
torch>=1.12.0
torchvision>=0.13.0
scikit-learn
matplotlib
numpy
```
