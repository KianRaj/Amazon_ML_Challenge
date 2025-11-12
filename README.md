# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** LightGBM 
**Team Members:** Dhruvkumar Patel, Meet Shah, Aman Kumar, Gour Dey
**Submission Date:** October 13, 2025

---

## 1. Executive Summary
Our solution employs a multimodal approach, integrating pre-computed text and image embeddings with engineered features to predict product prices. We utilize a LightGBM regressor trained on a feature set that has been refined through scaling and PCA for dimensionality reduction, leading to a robust and efficient model optimized for the SMAPE evaluation metric.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The core challenge lies in accurately predicting product prices from unstructured `catalog_content` and associated images. Our initial Exploratory Data Analysis (EDA) revealed several key points:
- The `catalog_content` is a semi-structured text field containing valuable explicit features like Item Pack Quantity (IPQ), numerical value, and unit of measure.
- The price distribution is heavily right-skewed, suggesting that a log-transformation of the target variable would be beneficial for model training.
- The high dimensionality of raw text and image embeddings requires effective feature selection or dimensionality reduction to prevent overfitting and reduce training time.

### 2.2 Solution Strategy
We adopted a single-model strategy centered around a highly optimized LightGBM Regressor, which is well-suited for handling the diverse and high-dimensional feature set.

**Approach Type:** Single Model (LightGBM) with Multimodal Feature Engineering.  
**Core Innovation:** Our main contribution is an efficient feature engineering pipeline that combines three distinct data sources:
1.  **Engineered Features:** Extracted structured data (IPQ, Value, Unit) from raw text using regex.
2.  **Textual Features:** High-dimensional embeddings from a powerful language model (`Qwen3-Embedding-0.6B`).
3.  **Visual Features:** High-dimensional embeddings from a state-of-the-art vision model (`dinov3-vits16-pretrain-lvd1689m`).

These features are then scaled and compressed using PCA before being fed into the LightGBM model, creating a powerful and computationally efficient solution.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Our architecture follows a sequential pipeline:
1.  **Feature Extraction:** Structured features (IPQ, Value, Unit) are parsed from the `catalog_content`.
2.  **Embedding Loading:** Pre-computed text and image embeddings are loaded from disk for each sample.
3.  **Feature Concatenation:** The engineered features, text embeddings, and image embeddings are combined into a single raw feature vector.
4.  **Dimensionality Reduction:** The high-dimensional embedding portions of the vector are independently scaled (`StandardScaler`) and then reduced using Principal Component Analysis (PCA) to capture the most significant variance in fewer dimensions.
5.  **Final Assembly:** The low-dimensional engineered features are concatenated with the PCA-reduced text and image features.
6.  **Model Training:** The final feature vector is used to train a `LGBMRegressor` on the log-transformed price, using SMAPE as the evaluation metric for early stopping.

### 3.2 Model Components

**Text Processing Pipeline:**
- **Preprocessing steps:**
    - Used pre-computed embeddings from `Qwen/Qwen3-Embedding-0.6B` (1024 dimensions).
    - Applied `StandardScaler` to the embeddings.
    - Applied `PCA` to reduce dimensionality from 1024 to 128.
- **Model type:** `Qwen3-Embedding-0.6B` for embedding generation.
- **Key parameters:** `n_components=128` for PCA.

**Image Processing Pipeline:**
- **Preprocessing steps:**
    - Used pre-computed embeddings from `facebook/dinov3-vits16` (384 dimensions).
    - Applied `StandardScaler` to the embeddings.
    - Applied `PCA` to reduce dimensionality from 384 to 64.
- **Model type:** `dinov3-vits16` for embedding generation.
- **Key parameters:** `n_components=64` for PCA.

**Regression Model:**
- **Model Type:** `LGBMRegressor`
- **Key Parameters:**
    - `n_estimators`: 5000 (with early stopping at 100 rounds)
    - `learning_rate`: 0.03
    - `num_leaves`: 40
    - `max_depth`: 7
    - `colsample_bytree`: 0.8
    - `subsample`: 0.8
    - `reg_alpha`: 0.1
    - `reg_lambda`: 0.1

---

## 4. Model Performance

### 4.1 Validation Results
The model was evaluated on a hold-out test set (15% of the training data) using the competition's official SMAPE metric.
- **SMAPE Score:** **[Enter your best validation SMAPE score from the script output here]**
- **Other Metrics:** The primary focus was optimizing for SMAPE. Other metrics like MAE/RMSE were not the primary objective for early stopping or hyperparameter tuning.

## 5. Conclusion
Our approach successfully integrates multimodal data into a powerful yet efficient gradient boosting framework. By combining deep learning embeddings with classic feature engineering and dimensionality reduction, we created a model that effectively captures the complex relationships between product attributes and price. The use of log-transformation and a custom SMAPE evaluation metric were critical to achieving a competitive score.

---

## Appendix

### A. Code Artefacts
- All Necessary Drive Link: `https://drive.google.com/drive/folders/1J_CSzrLex8B8uPmuaFZJcFHFBco5IfwX?usp=sharing`
- Training: `train.py`  
- Testing / Submission: `test.py`  
- Saved Objects: `tfidf_vectorizer.pkl`, `scaler.pkl`, `pca_model.pkl`  
- Saved Models: `best_fold_1.pt`, `best_fold_5.pt`  


### B. Additional Results
*The training script automatically generates feature caches (`.npy` files) to accelerate subsequent runs, demonstrating an efficient workflow for experimentation.*
*The `dataset_cleaning.py` script provides initial EDA, including distributions of key features like 'value' and 'price', which informed our decision to apply log-transformation and outlier removal.*

---

**Note:** This README is tailored for submission with the provided training and test scripts. The preprocessing objects and models must be included in the submission folder to replicate inference results.
