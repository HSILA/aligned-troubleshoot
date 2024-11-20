"""
This file will fetch the data from `Safawat/electrical_troubleshooting_data`
huggingface repository, which is a tabular data, extract the important fields and
saves each event as a separate .md file in ./config/kb directory which is the default 
directory for the knowledge base in nemo guardrails. This scirpt should be run before using this repo.
"""

import pandas as pd
import os

FILE_URL = 'https://huggingface.co/datasets/Safawat/electrical_troubleshooting_data/resolve/main/final.xlsx'

df = pd.read_excel(FILE_URL)

df.columns = [col.replace('\n', ' ').replace('/', '') for col in df.columns]
df.columns

df.dropna(subset=['Bar Size (mm)', 'Root Cause', 'Action Taken'], inplace=True)

os.makedirs('config/kb', exist_ok=True)

for i, row in df.iterrows():
    to_write = f"""
    Issue: {row['Consequence']}
    Bar Size (mm): {row['Bar Size (mm)']}
    Root Cause: {row['Root Cause']}
    Action Taken: {row['Action Taken']}
    """
    if pd.notna(row['Category Power Agency']):
        to_write += f"Category/Power Agency: {row['Category Power Agency']}\n"

    if pd.notna(row['Equipment  Name']):
        to_write += f"Equipment Name: {row['Equipment  Name']}\n"

    if pd.notna(row['Delay Type']) and row['Delay Type'] != 0 and row['Delay Type'] != np.nan:
        delay_type = row['Delay Type'].replace('\n', ' ')
        to_write += f"Delay Type: {delay_type}\n"

    if pd.notna(row['Category']):
        category = row['Category'].replace('\n', ' ')
        to_write += f"Delay Type: {category}\n"

    with open(os.path.join('config', 'kb', f'issue_{i}.md'), 'w', encoding='utf-8') as f:
        f.write(to_write)
