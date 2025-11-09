# Semantic Search for Anime

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A 12-week end-to-end machine learning project demonstrating industry-grade AI engineering skills. This project builds a semantic search and recommendation system for anime titles that understands natural language queries and retrieves relevant results using transformer embeddings and vector search.

## 🎯 Overview

Users can type natural-language queries like:
- *"Show me anime where humans fight gods"*
- *"Romantic comedies with a twist"*
- *"Space battles and mecha"*

The system understands **meaning** (not just keywords) and retrieves the most relevant anime using transformer embeddings and efficient vector search.

## ✨ Key Features

- **Natural Language Queries**: Search using conversational phrases instead of exact keywords
- **Transformer-Based Embeddings**: Leverages Sentence-BERT to generate semantic embeddings from:
  - Synopsis (plot summaries)
  - Genres (category tags)
  - Reviews (user feedback)
- **Weighted Embedding Fusion**: Customizable weights for different data sources (e.g., `0.5 * synopsis + 0.3 * genres + 0.2 * reviews`)
- **Fast Vector Search**: FAISS-powered similarity search for sub-200ms query latency
- **Flask Web Interface**: Simple, intuitive UI for interactive searching
- **Dynamic Results**: Top-ranked results with titles, genres, and synopses

## 🎓 Learning Objectives

This project demonstrates full-stack ML engineering skills:

- **Data Engineering**: Ingestion, cleaning, and preprocessing of Kaggle datasets
- **ML/NLP**: Text embeddings using Sentence-BERT transformers
- **Vector Databases**: FAISS index construction for semantic similarity search
- **Backend Development**: Flask/FastAPI for serving search results
- **DevOps**: Docker containerization and cloud deployment
- **MLOps**: Experiment tracking and monitoring (W&B / MLflow)

## 📊 Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision@5** | % of top 5 results that are relevant | ≥ 0.7 |
| **Latency** | Average response time | < 200 ms |
| **Recall@10** | Fraction of relevant items retrieved in top 10 | ≥ 0.9 |

## 📁 Project Structure

```
anime-semantic-search/
│
├── data/
│   ├── raw/                    # Original Kaggle CSVs
│   ├── clean/                  # Cleaned JSON/Parquet data
│   └── sample_queries.json     # Example test queries
│
├── src/
│   ├── ingest.py              # Cleans and merges raw CSVs
│   ├── embed.py               # Generates embeddings from synopses
│   ├── build_faiss.py         # Builds FAISS index for fast search
│   ├── search_api.py          # FastAPI endpoint /search?q=...
│   └── utils/                 # Helper functions
│
├── notebooks/
│   ├── Data_Exploration.ipynb
│   └── Evaluation.ipynb
│
├── embeddings/                 # Generated embedding files
│   ├── synopsis.npy
│   ├── genres.npy
│   └── reviews.npy
│
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

## 🛠️ Technology Stack

| Component | Tool |
|-----------|------|
| **Language** | Python 3.10+ |
| **ML/NLP** | sentence-transformers, torch |
| **Vector Search** | faiss-cpu |
| **API** | FastAPI, uvicorn, Flask |
| **Data** | pandas, pyarrow |
| **Deployment** | Docker, AWS/GCP (optional) |
| **MLOps** | W&B / MLflow |

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nsfogg/semantic-search.git
cd semantic-search
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file in the root directory:
```env
CLEAN_DATA_PATH=data/clean/anime.parquet
SNYOPSIS_EMBEDDINGS_PATH=embeddings/synopsis.npy
GENRES_EMBEDDINGS_PATH=embeddings/genres.npy
REVIEWS_EMBEDDINGS_PATH=embeddings/reviews.npy
```

### Usage

1. **Generate embeddings**
```bash
python src/embedding_encoder.py
```

2. **Run the Flask app**
```bash
python app.py
```

3. **Query the API**
```bash
curl "http://localhost:8000/search?q=anime about samurai fighting gods&k=5"
```

**Sample Response:**
```json
{
  "results": [
    {"title": "Bleach", "score": 0.91},
    {"title": "Dragon Ball Z", "score": 0.88},
    ...
  ]
}
```

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t anime-semantic-search .

# Run the container
docker run -p 8000:8000 anime-semantic-search
```

## 📈 Evaluation

Run the evaluation notebook to assess model performance:

```bash
jupyter notebook notebooks/Evaluation.ipynb
```

Key evaluation metrics:
- Precision and Recall at K
- Mean Average Precision (MAP)
- Query latency distribution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data © respective Kaggle contributors and MyAnimeList community
- Project inspired by modern vector-search systems used by Google, Spotify, and Netflix
- Built with [Sentence-BERT](https://www.sbert.net/) and [FAISS](https://github.com/facebookresearch/faiss)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating ML engineering best practices. The system is designed for learning purposes and may require optimization for production use.