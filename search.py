import pandas as pd
import pyarrow as pa
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def search(df, query, embeddings, weights, top_k=10):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query], normalize_embeddings=True)

    scores = np.zeros(embeddings[0].shape[0])
    for i in range(len(embeddings)):
        scores += weights[i] * cosine_similarity(query_vec, embeddings[i])[0]

    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return df.iloc[top_indices][['title']]

def load_embeddings():
    load_dotenv()
    df = pd.read_parquet(os.getenv("CLEAN_DATA_PATH"))
    e_synopsis = np.load(os.getenv("SNYOPSIS_EMBEDDINGS_PATH"))
    e_genres = np.load(os.getenv("GENRES_EMBEDDINGS_PATH"))
    e_reviews = np.load(os.getenv("REVIEWS_EMBEDDINGS_PATH"))

    return df, [e_synopsis, e_genres, e_reviews]

df, embeddings = load_embeddings()
prompt = "Becoming an adult and facing challenges and growing as a person"

res = search(df, prompt, embeddings, weights=[0.4, 0.2, 0.4])

print(res)
