import pandas as pd
import pyarrow as pa
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time
import joblib


# Load data
load_dotenv()
input_path = os.getenv("CLEAN_DATA_PATH")
print(input_path)
df = pd.read_parquet(input_path)
print(f"Loaded {len(df)} anime entries.\n", df.head())

model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and memory-efficient model

embeddings = model.encode(
    df['synopsis'].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(f"Embeddings shape: {embeddings.shape}")

def search(query, top_k=5):
    # Encode query
    query_vec = model.encode([query], normalize_embeddings=True)

    # Compute cosine similarities
    similarities = cosine_similarity(query_vec, embeddings)[0]

    # Get top k results
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    results = df.iloc[top_k_indices][['title', 'genres', 'score', 'synopsis']].copy()
    results['similarity'] = similarities[top_k_indices]

    return results

print("Testing search function...")
start = time.time()
res = search("A story about humans fighting gods")
end = time.time()
print(f"Query: 'A story about humans fighting gods':\n{res}\nTime taken: {end - start:.4f} seconds")

# Save model & embeddings
joblib.dump(model, "models/sbert_model.pkl")
np.save("data/clean/embeddings.npy", embeddings)
