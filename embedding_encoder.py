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

def fetch_and_append_reviews(df, save_path, checkpoint=100, start_at=-1):
    reviews_list = []
    start_index = 0
    save_path = os.getenv(save_path)

    try:
        progress_df = pd.read_parquet(save_path)
        start_index = len(progress_df)
        reviews_list = progress_df['reviews'].tolist()
        if start_at > -1:
            start_index = start_at
            reviews_list = reviews_list[:start_at]
            print(f"Reverting to index {start_index} and starting fresh from there.")
        else:
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

        if not reviews:
            reviews_list.append("")
        else:
            reviews_list.append(reviews)
        print(f"Fetched reviews for {title} (ID: {mal_id}) at index {index}")

        if (index + 1) % checkpoint == 0:
            df.loc[:index, 'reviews'] = reviews_list
            df.loc[:index].to_parquet(save_path, index=False)
            print(f"Checkpoint saved at index {index + 1}...")

        time.sleep(1) # Jikan rate limits / minute
    df['reviews'] = reviews_list
    df.to_parquet(save_path, index=False)
    print("All reviews fetched and saved.")
    return df

def get_reviews(anime_id, title, search_depth=100):
    res = ""
    page = 1
    reviews = []
    retries = 0
    max_retries = 5

    while len(reviews) < search_depth:
        try:
            print(f"Fetching reviews for {title} (ID: {anime_id}), page {page}...")
            response = requests.get(f"{base_url}/anime/{anime_id}/reviews",
                                    params={"page": page, "preliminary": "true", "spoiler": True},
                                    timeout=10)  # Timeout after 10 seconds
            response.raise_for_status()  # Raise an error for HTTP errors
            anime_data = response.json()
            reviews.extend(anime_data.get("data", []))
            if not anime_data.get("pagination", {}).get("has_next_page", False):
                break
            page += 1
            time.sleep(1)  # Wait 1 second between requests
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed to fetch reviews for {title} (ID: {anime_id}) after {max_retries} retries: {e}")
                break
            print(f"Retry {retries}/{max_retries} for {title} (ID: {anime_id}): {e}")
            time.sleep(5)  # Wait 5 seconds before retrying

    reviews = reviews[:search_depth]
    for i, review in enumerate(reviews, 1):
        res += f"{i}. {review['review']}\n\n"  # Add each review's text to the result

    return res

def encode_embeddings(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    synopsis = model.encode(
            df['synopsis'].tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    df['genreString'] = df['genres'].apply(lambda x: ', '.join(map(str, x)))
    genres = model.encode(
            df['genreString'].tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    reviews = model.encode(
            df['reviews'].tolist(),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    np.save(os.getenv("SNYOPSIS_EMBEDDINGS_PATH"), synopsis)
    np.save(os.getenv("GENRES_EMBEDDINGS_PATH"), genres)
    np.save(os.getenv("REVIEWS_EMBEDDINGS_PATH"), reviews)

df = load_data("CLEAN_DATA_PATH")
df = fetch_and_append_reviews(df, "REVIEWS_DATA_PATH", checkpoint=50)
encode_embeddings(df)