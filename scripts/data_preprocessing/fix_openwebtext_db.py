import pandas as pd
from tqdm import tqdm

from utils.constants import OUTPUT_DIR

doc_ids = pd.read_csv(OUTPUT_DIR / 'temp' / 'doc_ids.csv')
out_file = OUTPUT_DIR / 'temp' / 'spans_fixed.csv'
assert not out_file.exists()

filename = 'spans.csv'
CHUNK_SIZE = 10_000_000

attributes = [
    'toxicity',
    'severe_toxicity',
    'identity_attack',
    'insult',
    'threat',
    'profanity',
    'sexually_explicit',
    'flirtation'
]


def load_ids(old_df):
    old_df['location'] = old_df['filename'].apply(lambda x: x.split('.')[0])
    new_df = old_df.merge(doc_ids, on='location')
    assert len(new_df) == len(old_df)
    return new_df


if filename == 'docs.csv':
    total = 7770364
    columns = ['id'] + attributes

    print("Loading docs")
    df = pd.read_csv(OUTPUT_DIR / 'temp' / filename)
    out_df = load_ids(df)
    out_df.to_csv(out_file, header=True, index=False, columns=columns)
elif filename == 'spans.csv':
    total = 76221964
    columns = ['id', 'begin', 'end'] + attributes

    chunks = pd.read_csv(OUTPUT_DIR / 'temp' / filename, chunksize=CHUNK_SIZE)

    header = True
    chunk: pd.DataFrame
    for chunk in tqdm(chunks, total=total / CHUNK_SIZE):
        out_chunk = load_ids(chunk)
        out_chunk.to_csv(out_file, header=header, mode='a', index=False, columns=columns)
        header = False
else:
    raise RuntimeError
