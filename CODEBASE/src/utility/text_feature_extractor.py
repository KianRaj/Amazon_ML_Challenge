import os
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import concurrent.futures
import threading
from tqdm import tqdm

# Model name
model_name = "Qwen3-Embedding-0.6B"
pretrained_model_name = f"Qwen/{model_name}"

# Load data
df = pd.read_csv('../dataset/test.csv')

# Directories
embeddings_dir = Path('../embeddings/test') / model_name
embeddings_dir.mkdir(parents=True, exist_ok=True)
error_file = embeddings_dir / 'error_ids.txt'

# Global variables for threads (shared and thread-safe)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name)
device = torch.device("cuda:1")
model.to(device)
error_list = []
error_lock = threading.Lock()

def process_row(row):
    id_ = row.sample_id
    text = row.catalog_content
    emb_path = embeddings_dir / f"{id_}.npy"
    
    if emb_path.exists():
        return  # Already processed
    
    try:
        if pd.isna(text) or text.strip() == "":
            raise ValueError("Empty text")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        np.save(emb_path, embedding)
    except Exception as e:
        with error_lock:
            error_list.append(id_)

if __name__ == '__main__':
    # Number of threads
    max_workers = min(4, os.cpu_count())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Convert to list to avoid issues
        rows = list(df.itertuples(index=False))
        list(tqdm(executor.map(process_row, rows), total=len(df), desc="Processing texts"))
    
    # Write error ids
    with open(error_file, 'w') as f:
        for id_ in error_list:
            f.write(f"{id_}\n")
    
    print(f"Processing complete. Errors saved to {error_file}")