import pandas as pd
import os

# Path to your heavy file
file_path = "raw_data/india-news-headlines.csv"
chunk_size = 1000000  # 1 million rows per file is roughly 75-85 MB

print("Commencing dataset fragmentation...")
for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
    output_name = f"raw_data/india-news-headlines_part_{i+1}.csv"
    chunk.to_csv(output_name, index=False)
    print(f"Successfully generated: {output_name}")

print("Fragmentation complete. Ensure the original 271MB file is ignored by Git!")