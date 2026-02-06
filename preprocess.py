import pandas as pd
from datasets import load_dataset
import re

print("STEP 1: FIXED PREPROCESSING")
print("---------------------------")

# 1. Load Data
dataset = load_dataset("ShenLab/MentalChat16K", split="train")
df = dataset.to_pandas()
print(f"Original Count: {len(df)}")

# 2. Inspect Columns (Critical Check)
# We fill empty 'input' rows with 'instruction' just in case some rows are swapped.
df['input'] = df['input'].fillna('')
df['user_message'] = df.apply(lambda row: row['input'] if len(row['input']) > 5 else row['instruction'], axis=1)

# 3. Cleaning Function
def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text).strip()

df['user_message'] = df['user_message'].apply(clean_text)
df['output'] = df['output'].apply(clean_text)

# 4. Filter Garbage
# Deduplicate based on the RESPONSE (output), because every good response should be unique.
df.drop_duplicates(subset=['output'], inplace=True)

# Remove generic short inputs
df = df[df['user_message'].str.len() > 10]
df.dropna(subset=['user_message', 'output'], inplace=True)

# 5. Save
# We rename 'user_message' to 'instruction' so it fits the next script perfectly
df = df.rename(columns={'user_message': 'final_instruction'})
final_df = df[['final_instruction', 'output']]

output_csv = "MentalChat16K_Cleaned.csv"
final_df.to_csv(output_csv, index=False)

print(f"Final Cleaned Count: {len(final_df)}")
print(f"Saved to '{output_csv}'")