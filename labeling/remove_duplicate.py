#clayzzz

import pandas as pd
from tqdm import tqdm

def remove_duplicates_fast_with_progress(input_csv, output_csv, chunksize=100000 # using chunk -> 100 000 lines per chunk):
    print(f"📂 Reading {input_csv} per chunk ...")
    reader = pd.read_csv(input_csv, chunksize=chunksize)

    chunks = []
    total_rows = 0
    
    # loop through csv per "chunk" each time
    for chunk in tqdm(reader, desc="⏳ Read file (chunk)", dynamic_ncols=True):
        
        chunks.append(chunk)
        # remember lines were read
        total_rows += len(chunk)

    print(f"🔍 Total lines: {total_rows:,}")
    print("🔍 Removing duplicates...")

    df = pd.concat(chunks, ignore_index=True) # used concat to connect seperates chunks
    df_unique = df.drop_duplicates(subset=["password"]) # remove duplicate by pandas, subset == column in csv

    print(f"✅ Result: {len(df_unique):,}")
    print(f"💾 Saving file to: {output_csv} ...")
    df_unique.to_csv(output_csv, index=False) # extract to csv file
    print("✅ Complete!")
    

input_csv = "..." # you CSV path
output_csv = "..." # output path and csv name

remove_duplicates_fast_with_progress(input_csv, output_csv)
