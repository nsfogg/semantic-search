# 🎌 Anime Semantic Search
*A 12-week end-to-end machine learning project demonstrating industry-grade AI engineering skills.*

---

## 📘 Overview
This project builds a **semantic search and recommendation system** for anime titles.  
Users can type natural-language queries like:

> “Show me anime where humans fight gods.”

The system understands meaning (not just keywords) and retrieves the most relevant shows using transformer embeddings and vector search.

---

## 🧭 Project Goals
- Demonstrate full-stack ML engineering skills:
  - Data ingestion and cleaning
  - Text embeddings (Sentence-BERT)
  - Vector similarity search (FAISS)
  - FastAPI service for search
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

## 🧱 Week 1: Data Ingestion & Schema
### Dataset
Data sourced from [Kaggle Anime Dataset](https://www.kaggle.com/datasets) containing:
- `title` — unique anime name  
- `synopsis` — plot summary text  
- `genres` — comma-separated list of tags  
- `rating` — numeric user rating  

### Cleaning Steps
1. Rename columns to standardized names.
2. Drop rows missing `title` or `synopsis`.
3. Convert `genres` into list format (`["Action", "Drama"]`).
4. Deduplicate entries by title.
5. Export unified dataset to `data/clean/clean_anime.json`.

Example record:
```json
{
  "anime_id": 101,
  "title": "Attack on Titan",
  "synopsis": "After his hometown is destroyed...",
  "genres": ["Action", "Drama", "Fantasy"],
  "rating": 8.9
}
```

### Run the ingestion script
```bash
python src/ingest.py
```
**Output:**
```
Saved 12,543 cleaned anime records → data/clean/clean_anime.json
```

---

## ⚙️ Next Steps
| Phase | Description | Deliverable |
|-------|--------------|-------------|
| Week 2 | Convert synopses to vector embeddings using Sentence-BERT | `embeddings.npy` |
| Week 3 | Build FAISS index for fast similarity search | `faiss.index` |
| Week 4 | Expose `/search` API via FastAPI | Local demo |

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

## 🧠 Credits & Inspiration
Data © respective Kaggle contributors and MyAnimeList community.  
Project inspired by modern vector-search systems used by Google, Spotify, and Netflix.

---

## 📅 Status
**Current phase:** Week 1 — Data ingestion and schema design.  
**Next:** Generate embeddings for semantic search.

