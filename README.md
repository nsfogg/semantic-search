# 🎌 Anime Semantic Search
*A 12-week end-to-end machine learning project demonstrating industry-grade AI engineering skills.*

---

## 📘 Overview
This project builds a **semantic search and recommendation system** for anime titles.  
Users can type natural-language queries like:

> “Show me anime where humans fight gods.”

The system understands meaning (not just keywords) and retrieves the most relevant shows using transformer embeddings and vector search.

The project includes a **web-based frontend** for users to interact with the search system, powered by Flask and hosted locally or in the cloud.

---

## 🧭 Project Goals
- Demonstrate full-stack ML engineering skills:
  - Data ingestion and cleaning
  - Text embeddings (Sentence-BERT)
  - Vector similarity search (FAISS)
  - Flask-based web frontend for search
  - Docker + Cloud deployment
  - Experiment tracking and monitoring (W&B / MLflow)
- Produce measurable metrics:
  - **Precision@5 ≥ 0.7**
  - **Average query latency < 200 ms**

---

## 🗂️ Repository Structure
```
anime-semantic-search/
│
├── data/
│   ├── raw/                # Original Kaggle CSVs
│   ├── clean/              # Cleaned JSON / Parquet data
│   └── sample_queries.json # Example test queries
│
├── src/
│   ├── ingest.py           # Cleans and merges raw CSVs → unified dataset
│   ├── embed.py            # Generates embeddings from synopses
│   ├── build_faiss.py      # Builds FAISS index for fast semantic search
│   ├── search_api.py       # FastAPI endpoint /search?q=...
│   └── utils/              # Helper functions
│
├── notebooks/
│   ├── Data_Exploration.ipynb
│   └── Evaluation.ipynb
│
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

---

## 🧩 Tech Stack
| Component | Tool |
|------------|------|
| Language | Python 3.10+ |
| ML | `sentence-transformers`, `torch` |
| Vector Search | `faiss-cpu` |
| API | `FastAPI`, `uvicorn` |
| Data | `pandas`, `pyarrow` |
| Deployment | Docker, (AWS/GCP optional) |

---

## 🧪 Example Usage (later phase)
```bash
curl "http://localhost:8000/search?q=anime about samurai fighting gods&k=5"
```
**Response:**
```json
{
  "results": [
    {"title": "Bleach", "score": 0.91},
    {"title": "Dragon Ball Z", "score": 0.88},
    ...
  ]
}
```

---

## 📊 Evaluation Metrics
| Metric | Description | Target |
|---------|--------------|--------|
| Precision@5 | % of top 5 results that are relevant | ≥ 0.7 |
| Latency | Average response time | < 200 ms |
| Recall@10 | Fraction of relevant items retrieved | ≥ 0.9 |

---

## 🛠️ Features
### 🔍 Semantic Search
- **Natural Language Queries**: Users can type queries like "anime about space battles" or "romantic comedies with a twist."
- **Transformer-Based Embeddings**: Uses `Sentence-BERT` to generate embeddings for:
  - **Synopsis**: The plot summary of the anime.
  - **Genres**: The genre tags associated with the anime.
  - **Reviews**: User reviews for the anime.

### ⚖️ Weighted Search
- Combines embeddings for `synopsis`, `genres`, and `reviews` using customizable weights:
  - Example: `0.5 * synopsis + 0.3 * genres + 0.2 * reviews`
- Allows fine-tuning of search relevance based on user preferences.

### 🌐 Web Frontend
- **Flask-Based Web App**: A simple and intuitive interface for users to interact with the search system.
- **Dynamic Results**: Displays the top search results with titles, genres, and synopses.

---

## 🚀 How to Run the Project
### 1. Clone the Repository
```bash
git clone https://github.com/nsfogg/semantic-search.git
cd semantic-search
```

### 2. Install Dependencies
Install required dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a .env file in the root directory with the following variables:
```json
CLEAN_DATA_PATH=data/clean/anime.parquet
SNYOPSIS_EMBEDDINGS_PATH=embeddings/synopsis.npy
GENRES_EMBEDDINGS_PATH=embeddings/genres.npy
REVIEWS_EMBEDDINGS_PATH=embeddings/reviews.npy
```

### 4. Generate Embeddings
Run the embedding_encoder.py script to generate embeddings

### 5. Start the Web App
Run the Flask app

---

## 🧠 Credits & Inspiration
Data © respective Kaggle contributors and MyAnimeList community.  
Project inspired by modern vector-search systems used by Google, Spotify, and Netflix.
