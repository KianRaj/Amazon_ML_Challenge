import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib
import time
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor, early_stopping

# ==== CONFIGURATION ====
# Use a smaller subset for faster testing by setting a number, e.g., 10000
DATA_SAMPLE_SIZE = None # Set to None to use the full dataset
CSV_PATH = "../dataset/train.csv"
TEXT_EMB_DIR = "embeddings/train/Qwen3-Embedding-0.6B"
IMAGE_EMB_DIR = "embeddings/train/dinov3-vits16-pretrain-lvd1689m"
MODEL_PATH = "price_predictor_lgbm_v2.pkl"
SCALER_IMG_PATH = "scaler_img_v2.pkl"
SCALER_TXT_PATH = "scaler_txt_v2.pkl"
PCA_IMG_PATH = "pca_img_v2.pkl"
PCA_TXT_PATH = "pca_txt_v2.pkl"
UNIT_ENCODER_PATH = "unit_encoder_v2.pkl"
X_CACHE = "X_combined_v4.npy"
Y_CACHE = "y_log_v4.npy"

# Embedding dimensions should match your pre-computed files
IMG_DIM = 384  # For facebook/dinov3-vits16
TXT_DIM = 1024 # For Qwen/Qwen3-Embedding-0.6B
NUM_EXTRA_FEATURES = 3 # IPQ, Value, Unit

# ==== HELPER & FEATURE EXTRACTION FUNCTIONS ====
def extract_ipq(catalog_content: str) -> int:
    """Extracts Item Pack Quantity (IPQ) using a comprehensive set of regex patterns."""
    if not isinstance(catalog_content, str): return 1
    patterns = [
        r'\(Pack of (\d+)\)', r'pack of (\d+)', r'(\d+)\s*-\s*pack', r'(\d+)\s*packs',
        r'case of (\d+)', r'set of (\d+)', r'(\d+)\s*count', r'(\d+)\s*ct',
        r'(\d+)\s*x\s', r'includes (\d+) boxes', r'(\d+)\s*cans', r'(\d+)\s*bottles'
    ]
    for pattern in patterns:
        match = re.search(pattern, catalog_content, re.IGNORECASE)
        if match:
            try: return int(match.group(1))
            except (ValueError, IndexError): continue
    return 1

def extract_value(catalog_content: str) -> float:
    """Extracts the 'Value' (e.g., 72.0) from the text."""
    if not isinstance(catalog_content, str): return 0.0
    match = re.search(r'Value:\s*([\d.]+)', catalog_content, re.IGNORECASE)
    if match:
        try: return float(match.group(1))
        except (ValueError, IndexError): return 0.0
    return 0.0

def extract_unit(catalog_content: str) -> str:
    """Extracts the 'Unit' (e.g., 'Fl Oz') from the text."""
    if not isinstance(catalog_content, str): return 'unknown'
    match = re.search(r'Unit:\s*([\w\s]+)', catalog_content, re.IGNORECASE)
    if match:
        unit_str = match.group(1).strip()
        return unit_str.split('\n')[0].strip()
    return 'unknown'

def smape(y_true, y_pred):
    """Calculates SMAPE as a percentage."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.maximum(denominator, 1e-9)
    return np.mean(diff) * 100

def lgbm_smape(y_true, y_pred):
    """Custom SMAPE for LightGBM, operating on log-transformed values."""
    y_true_inv = np.expm1(y_true)
    y_pred_inv = np.expm1(y_pred)
    score = smape(y_true_inv, y_pred_inv)
    return 'smape', score, False # Lower is better

# ==== FEATURE LOADING & BUILDING ====
def build_or_load_features(df):
    if os.path.exists(X_CACHE) and os.path.exists(Y_CACHE):
        print("⚡ Loading precomputed feature matrix from cache...")
        X = np.load(X_CACHE)
        y = np.load(Y_CACHE)
        print(f"✅ Cached data loaded: X={X.shape}, y={y.shape}")
        return X, y

    print("📦 Building feature matrix (this might take a while)...")
    
    print("🔨 Pre-extracting text features (IPQ, Value, Unit)...")
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['value'] = df['catalog_content'].apply(extract_value)
    df['unit'] = df['catalog_content'].apply(extract_unit)

    unique_units = df['unit'].unique()
    unit_to_int = {unit: i for i, unit in enumerate(unique_units)}
    joblib.dump(unit_to_int, UNIT_ENCODER_PATH)
    print(f"Found {len(unique_units)} unique units. Encoded and saved mapping.")
    df['unit_encoded'] = df['unit'].map(unit_to_int)

    def load_single_emb(file_path, target_dim):
        try:
            emb = np.load(file_path).flatten().astype(np.float32)
            if len(emb) == target_dim: return file_path.stem, emb
            elif len(emb) > target_dim: return file_path.stem, emb[:target_dim]
            else: return file_path.stem, np.pad(emb, (0, target_dim - len(emb)), 'constant')
        except Exception: return file_path.stem, None

    with ThreadPoolExecutor(max_workers=12) as executor:
        img_files = list(Path(IMAGE_EMB_DIR).glob("*.npy"))
        img_results = list(tqdm(executor.map(lambda f: load_single_emb(f, IMG_DIM), img_files), total=len(img_files), desc="Loading image embeddings"))
        img_embeddings = {img_id: emb for img_id, emb in img_results if emb is not None}
        
        txt_files = list(Path(TEXT_EMB_DIR).glob("*.npy"))
        txt_results = list(tqdm(executor.map(lambda f: load_single_emb(f, TXT_DIM), txt_files), total=len(txt_files), desc="Loading text embeddings"))
        txt_embeddings = {txt_id: emb for txt_id, emb in txt_results if emb is not None}
        
    features, targets = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building final feature rows"):
        sample_id = str(row["sample_id"])
        feat_vector = [row['ipq'], row['value'], row['unit_encoded']]
        
        img_emb = img_embeddings.get(sample_id, np.zeros(IMG_DIM, dtype=np.float32))
        txt_emb = txt_embeddings.get(sample_id, np.zeros(TXT_DIM, dtype=np.float32))
        
        feat = np.concatenate([feat_vector, img_emb, txt_emb])
        features.append(feat)
        targets.append(np.log1p(row["price"])) # Log-transform the price
        
    X_raw = np.array(features, dtype=np.float32)
    y_log = np.array(targets, dtype=np.float32)

    np.save(X_CACHE, X_raw)
    np.save(Y_CACHE, y_log)
    print(f"✅ Features cached: X={X_raw.shape}, y={y_log.shape}")
    return X_raw, y_log

# ==== MAIN SCRIPT ====
if __name__ == "__main__":
    print("="*60)
    print("🚀 AMAZON ML PRICE PREDICTOR - TRAINING SCRIPT")
    print("="*60)

    full_df = pd.read_csv(CSV_PATH)
    df = full_df.sample(n=DATA_SAMPLE_SIZE, random_state=42) if DATA_SAMPLE_SIZE else full_df
    
    X_raw, y_log = build_or_load_features(df)
    X_train_raw, X_test_raw, y_train_log, y_test_log = train_test_split(X_raw, y_log, test_size=0.15, random_state=42)

    print("\n✨ Scaling Embeddings and Applying PCA (No Data Leakage)...")
    num_features_train = X_train_raw[:, :NUM_EXTRA_FEATURES]
    img_emb_train = X_train_raw[:, NUM_EXTRA_FEATURES : NUM_EXTRA_FEATURES+IMG_DIM]
    txt_emb_train = X_train_raw[:, NUM_EXTRA_FEATURES+IMG_DIM:]
    
    num_features_test = X_test_raw[:, :NUM_EXTRA_FEATURES]
    img_emb_test = X_test_raw[:, NUM_EXTRA_FEATURES : NUM_EXTRA_FEATURES+IMG_DIM]
    txt_emb_test = X_test_raw[:, NUM_EXTRA_FEATURES+IMG_DIM:]

    scaler_img = StandardScaler().fit(img_emb_train)
    img_emb_train_s, img_emb_test_s = scaler_img.transform(img_emb_train), scaler_img.transform(img_emb_test)
    joblib.dump(scaler_img, SCALER_IMG_PATH)

    scaler_txt = StandardScaler().fit(txt_emb_train)
    txt_emb_train_s, txt_emb_test_s = scaler_txt.transform(txt_emb_train), scaler_txt.transform(txt_emb_test)
    joblib.dump(scaler_txt, SCALER_TXT_PATH)
    
    pca_img = PCA(n_components=64, random_state=42).fit(img_emb_train_s)
    img_emb_train_pca, img_emb_test_pca = pca_img.transform(img_emb_train_s), pca_img.transform(img_emb_test_s)
    joblib.dump(pca_img, PCA_IMG_PATH)
    
    pca_txt = PCA(n_components=128, random_state=42).fit(txt_emb_train_s)
    txt_emb_train_pca, txt_emb_test_pca = pca_txt.transform(txt_emb_train_s), pca_txt.transform(txt_emb_test_s)
    joblib.dump(pca_txt, PCA_TXT_PATH)
    
    X_train = np.hstack([num_features_train, img_emb_train_pca, txt_emb_train_pca])
    X_test = np.hstack([num_features_test, img_emb_test_pca, txt_emb_test_pca])
    
    print(f"\n📊 Final Feature Shapes: Train={X_train.shape}, Test={X_test.shape}")

    print("\n🤖 Training LightGBM Model...")
    model = LGBMRegressor(
        n_estimators=5000, learning_rate=0.03, num_leaves=40, max_depth=7,
        random_state=42, n_jobs=-1, colsample_bytree=0.8, subsample=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
    )

    start_train = time.time()
    model.fit(
        X_train, y_train_log, eval_set=[(X_test, y_test_log)],
        eval_metric=lgbm_smape, callbacks=[early_stopping(100, verbose=True)]
    )
    print(f"✅ Training completed in {(time.time() - start_train):.2f}s")
    
    print("\n🔍 Evaluation on Test Set:")
    y_pred_log_test = model.predict(X_test)
    y_test_actual = np.expm1(y_test_log)
    y_pred_actual_test = np.expm1(y_pred_log_test)
    y_pred_actual_test[y_pred_actual_test < 0] = 0 # Ensure non-negative prices
    
    test_smape_score = smape(y_test_actual, y_pred_actual_test)
    print(f"   SMAPE Score:  {test_smape_score:.4f}%")

    print(f"\n💾 Saving model and transformers to disk...")
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    print("\n" + "="*60 + "\n✨ Training Complete!\n" + "="*60)