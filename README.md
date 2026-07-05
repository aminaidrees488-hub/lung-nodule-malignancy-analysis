# Study of Machine Learning Models for Lung Nodule Malignancy

This repository contains a synthetic dataset developed for a machine learning project on lung nodule classification. 

## 📌 Project Overview
This dataset was created for a machine learning project on lung nodule classification. It contains clinical, demographic, radiomic, and fractal features collected to classify lung nodules as Benign or Malignant. The dataset can be used for exploratory data analysis, feature engineering, machine learning model development, and classification performance evaluation.
## 📊 Dataset Specifications
The primary data is hosted in the structured repository file: **`Final_Lung_Nodule_Data_Updated (1).csv`**

- **Cohort Size:** $N = 1000$ distinct structural records.
- **Dimensionality:** 16 independent predictive attributes and 1 target variable.
- **Target Variable:** `Malignancy_Grade` (Categorized on a clinical severity scale from 1 to 5).

### Feature Domains Included:
1. **Patient Demographics & History:** Age, Gender, Smoking History (Years), Family History, and Previous Malignancy status.
2. **Geometric Features:** Nodule Diameter (mm), Area, Perimeter, Sphericity, Convexity, and Edge Type (Smooth, Lobulated, Spiculated).
3. **Radiomic Texture & Fractal Descriptors:** Mean Intensity, Standard Deviation, Entropy, Box-Counting Fractal Dimension ($FD_{bc}$), and Power Spectrum Fractal Dimension ($FD_{ps}$).

## 🚀 Quick Start & Data Loading
Use the following Python code to load the dataset into a Pandas DataFrame for analysis and machine learning tasks.
```python
import pandas as pd

# Load dataset
df = pd.read_csv("Final_Lung_Nodule_Data_Updated (1).csv")

# Display dataset shape
print(f"Dataset loaded successfully! Shape: {df.shape}")

# Preview the first five rows
df.head()
