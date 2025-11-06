import pandas as pd
import pyarrow as pa
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import heapq
import requests

base_url = "https://api.jikan.moe/v4"

def load_data(path):
    load_dotenv()
    input = os.getenv(path)
    df = pd.read_parquet(input)
    return df

def fetch_and_append_reviews(df, save_path, checkpoint=100):
    reviews_list = []
    start_index = 0
    save_path = os.getenv(save_path)

    try:
        progress_df = pd.read_parquet(save_path)
        start_index = len(progress_df)
        reviews_list = progress_df['reviews'].tolist()
        print(f"Resuming from index {start_index}")
    except Exception as e:
        print("No existing progress file found. Starting fresh.")

    print(f"DataFrame length: {len(df)}")
    print(f"Starting index: {start_index}")
    print(f"Rows to process: {len(df.iloc[start_index:])}")
    for index, row in df.iloc[start_index:].iterrows():
        mal_id = row['malId']
        title = row['title']

        # Use a while loop to retry fetching reviews until successful
        while True:
            try:
                reviews = get_reviews(anime_id=mal_id, title=title)
                break  # Exit the loop if successful
            except Exception as e:
                print(f"Failed to fetch reviews for {title} (ID: {mal_id}): {e}")
                print("Retrying...")
                time.sleep(5)  # Wait 5 seconds before retrying

        reviews_list.append(reviews)

        if (index + 1) % checkpoint == 0:
            df.loc[:index, 'reviews'] = reviews_list
            df.loc[:index].to_parquet(save_path, index=False)
            print(f"Checkpoint saved at index {index + 1}...")

        time.sleep(1) # Jikan rate limits / minute
    df['reviews'] = reviews_list
    df.to_parquet(save_path, index=False)
    print("All reviews fetched and saved.")
    return df

def get_reviews(anime_id, title, num_reviews=2, search_depth=100):
    res = f"Title: {title}\n\nReviews:\n"
    page = 1
    reviews = []
    while len(reviews) < search_depth:
        response = requests.get(f"{base_url}/anime/{anime_id}/reviews",
                                params={"page": page, "preliminary": "true", "spoiler": True})
        if response.status_code != 200:
            print(f"Failed to retrieve data: {response.status_code}")
            break
        anime_data = response.json()
        reviews.extend(anime_data.get("data", []))
        if not anime_data.get("pagination", {}).get("has_next_page", False):
            break
        page += 1

    reviews = reviews[:search_depth]

    heap = []
    for review in reviews:
        impressions = review['reactions']['overall']  # Assuming "votes" represents impressions
        if len(heap) < num_reviews:
            heapq.heappush(heap, (impressions, review))
        else:
            heapq.heappushpop(heap, (impressions, review))

    # Extract the top reviews from the heap and sort them by impressions (descending)
    top_reviews = sorted(heap, key=lambda x: x[0], reverse=True)

    for i, (impressions, review) in enumerate(top_reviews, 1):
        res += f"{i}. {review['review'][:900]}\n\n" # Limit review length for clarity
    return res

def encode_embeddings(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
            df['synopsis'].tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    np.save(os.getenv("EMBEDDINGS_PATH"), embeddings)

df = load_data("CLEAN_DATA_PATH")
df = fetch_and_append_reviews(df, "REVIEWS_DATA_PATH", checkpoint=50)
encode_embeddings(df)