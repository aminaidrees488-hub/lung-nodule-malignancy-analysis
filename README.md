# Study of Machine Learning Models for Lung Nodule Malignancy

This repository contains the official dataset and programmatic framework for the statistical evaluation and machine learning-based classification of lung nodule malignancy risk. 

## 📌 Project Overview
The objective of this study is to implement and benchmark robust binary classification models to predict the malignancy grade of lung nodules. By integrating classical clinical risk indicators with advanced computational morphology and radiomic texture features, this framework aims to assist in the early and accurate assessment of pulmonary oncological risks.

## 📊 Dataset Specifications
The primary data is hosted in the structured repository file: **`Final_Lung_Nodule_Data_Updated (1).csv`**

- **Cohort Size:** $N = 1000$ distinct structural records.
- **Dimensionality:** 16 independent predictive attributes and 1 target variable.
- **Target Variable:** `Malignancy_Grade` (Categorized on a clinical severity scale from 1 to 5).

### Feature Domains Included:
1. **Patient Demographics & History:** Age, Gender, Smoking History (Years), Family History, and Previous Malignancy status.
2. **Morphological & Geometric Features:** Nodule Diameter (mm), Area, Perimeter, Sphericity, Convexity, and Edge Type (Smooth, Lobulated, Spiculated).
3. **Radiomic Texture & Fractal Descriptors:** Mean Intensity, Standard Deviation, Entropy, Box-Counting Fractal Dimension ($FD_{bc}$), and Power Spectrum Fractal Dimension ($FD_{ps}$).

## 🚀 Quick Start & Data Loading
To replicate the exploratory data analysis or build machine learning pipelines (e.g., Random Forest, Gradient Boosting, Gaussian Naive Bayes), you can load the dataset using the following Python snippet:

```python
import pandas as pd

# Load dataset
df = pd.read_csv("Final_Lung_Nodule_Data_Updated (1).csv")

# Verify integrity
print(f"Data Loaded Successfully. Shape: {df.shape}")
