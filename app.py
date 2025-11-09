from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load embeddings and data
def load_embeddings():
    load_dotenv()
    df = pd.read_parquet(os.getenv("CLEAN_DATA_PATH"))
    e_synopsis = np.load(os.getenv("SNYOPSIS_EMBEDDINGS_PATH"))
    e_genres = np.load(os.getenv("GENRES_EMBEDDINGS_PATH"))
    e_reviews = np.load(os.getenv("REVIEWS_EMBEDDINGS_PATH"))
    return df, [e_synopsis, e_genres, e_reviews]

df, embeddings = load_embeddings()

# Search function
def search(df, query, embeddings, weights, top_k=10):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query], normalize_embeddings=True)

    scores = np.zeros(embeddings[0].shape[0])
    for i in range(len(embeddings)):
        scores += weights[i] * cosine_similarity(query_vec, embeddings[i])[0]

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_indices][['title', 'genres', 'synopsis']]

    # Ensure all values are JSON-serializable
    results = results.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    results = results.applymap(lambda x: str(x) if not isinstance(x, (str, list, dict)) else x)

    return results.to_dict(orient='records')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_route():
    data = request.json
    query = data.get('query', '')
    weights = data.get('weights', [0.4, 0.2, 0.4])
    top_k = data.get('top_k', 10)

    results = search(df, query, embeddings, weights, top_k)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)