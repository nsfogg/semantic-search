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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from openai import OpenAI

# Load data
load_dotenv()
input_path = os.getenv("CLEAN_DATA_PATH")
print(input_path)
df = pd.read_parquet(input_path)
print(f"Loaded {len(df)} anime entries.\n", df.head())

model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and memory-efficient model

# embeddings = model.encode(
#     df['synopsis'].tolist(),
#     batch_size=64,
#     show_progress_bar=True,
#     convert_to_numpy=True,
#     normalize_embeddings=True
# )
# print(f"Embeddings shape: {embeddings.shape}")

# Check if embeddings already exist
if os.path.exists(os.getenv("EMBEDDINGS_PATH")):
    print("Loading precomputed embeddings...")
    embeddings = np.load(os.getenv("EMBEDDINGS_PATH"))
else:
    print("Computing embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and memory-efficient model
    embeddings = model.encode(
        df['synopsis'].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"Embeddings shape: {embeddings.shape}")
    # Save embeddings for future use
    np.save(os.getenv("EMBEDDINGS_PATH"), embeddings)
    # Optionally save the model
    joblib.dump(model, "models/sbert_model.pkl")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def improve_query(query):
    prompt = f"Improve the following search query to be more descriptive and specific:\n\nOriginal Query: '{query}'"
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that improves search queries about searching for anime. You should expand the query to make it more specific and descriptive in regards to anime."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.7
    )
    improved_query = response.choices[0].message.content.strip()
    return improved_query

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

# TF-IDF Vectorizer for comparison
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['synopsis'])

# Hybrid search using both semantic and keyword matching
def hybrid_search(query, top_k=5, alpha=0.8):
    query_vec = model.encode([query], normalize_embeddings=True)
    semantic_sims = cosine_similarity(query_vec, embeddings)[0]

    tfidf_query = tfidf.transform([query])
    tfidf_sims = (tfidf_matrix @ tfidf_query.T).toarray().ravel()

    # Combine semantic + keyword
    scores = alpha * semantic_sims + (1 - alpha) * normalize(tfidf_sims.reshape(1, -1))[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return df.iloc[top_indices][['title', 'genres', 'synopsis']], scores[top_indices]

prompt = "Fantasy focused on adventure and growth"

print("Testing search function...")
start = time.time()
res = search(prompt)
end = time.time()
print(f"Query: '{prompt}':\n{res}\nTime taken: {end - start:.4f} seconds")

# start_v2 = time.time()
# improved_prompt = improve_query(prompt)
# res_v2 = hybrid_search(improved_prompt)
# end_v2 = time.time()
# print(f"Improved Query: '{improved_prompt}':\n{res_v2}\nTime taken: {end_v2 - start_v2:.4f} seconds")

# Save model & embeddings
# joblib.dump(model, "models/sbert_model.pkl")
# np.save("data/clean/embeddings.npy", embeddings)
