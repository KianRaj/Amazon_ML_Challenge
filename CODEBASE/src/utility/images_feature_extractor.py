import os
import pandas as pd
from pathlib import Path
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from huggingface_hub import login
import concurrent.futures
import threading
from tqdm import tqdm

# Log in to Hugging Face Hub with your token
login(token="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")

pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
model_name = pretrained_model_name.split('/')[-1]

# Load data
df = pd.read_csv('../dataset/test.csv')

# Directories
images_dir = Path('../images/test')
embeddings_dir = Path('../embeddings/test') / model_name
embeddings_dir.mkdir(parents=True, exist_ok=True)
error_file = embeddings_dir / 'error_ids.txt'

# Global variables for threads (shared and thread-safe)
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name)
device = torch.device("cuda:1")  # Use CUDA device 1 for parallel processing
model.to(device)
error_list = []
error_lock = threading.Lock()

def process_row(row):
    id_ = row.sample_id  # Changed from row.id
    url = row.image_link  # Changed from row.image_url
    image_name = url.split('/')[-1]
    image_path = images_dir / image_name
    emb_path = embeddings_dir / f"{id_}.npy"
    
    if emb_path.exists():
        return  # Already processed
    
    try:
        image = load_image(str(image_path))
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        embedding = outputs.pooler_output.cpu().numpy()
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
        list(tqdm(executor.map(process_row, rows), total=len(df), desc="Processing images"))
    
    # Write error ids
    with open(error_file, 'w') as f:
        for id_ in error_list:
            f.write(f"{id_}\n")
    
    print(f"Processing complete. Errors saved to {error_file}")