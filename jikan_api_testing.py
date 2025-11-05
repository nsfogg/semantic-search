import requests
import json
import heapq
import time

base_url = "https://api.jikan.moe/v4"

anime_id = 1
# response = requests.get(f"{base_url}/anime/{anime_id}")

# if response.status_code == 200:
#     anime_data = response.json()
#     print(json.dumps(anime_data, indent=4))
# else:
#     print(f"Failed to retrieve data: {response.status_code}")

def get_anime(anime_id):
    response = requests.get(f"{base_url}/anime/{anime_id}")
    if response.status_code == 200:
        anime_data = response.json()
        return anime_data['data']['title']
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

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

# x = get_reviews(anime_id=1, title="Cowboy Bebop")

for i in range(1, 10):
    anime = get_anime(i)
    print(f"Anime ID: {i}, Title: {anime}")
    time.sleep(1)
