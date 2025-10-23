import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
import os

load_dotenv()

csv_file_path = os.getenv("RAW_DATA_PATH")
df = pd.read_csv(csv_file_path)

# Clean
df = df.rename(columns={"MAL_ID": "malId", "Name": "title", "Score": "score", "Genres": "genres", "sypnopsis": "synopsis"})
df = df.dropna()
df = df.drop_duplicates(subset=["title"])
df = df[df['synopsis'] != 'No synopsis information has been added to this title. Help improve our database by adding a synopsis here .']
df["genres"] = df["genres"].apply(lambda g: [x.strip() for x in g.split(",")] if isinstance(g, str) else [])
print(f"Cleaned {len(df)} anime records.")

table = pa.Table.from_pandas(df)

parquet_file_path = os.getenv("CLEAN_DATA_PATH")
pq.write_table(table, parquet_file_path)
