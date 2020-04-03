import pickle
from multiprocessing.pool import Pool

import pandas as pd
import spacy
from tqdm.auto import tqdm

from utils.constants import TEXTS_DIR, OUTPUT_DIR
from utils.db import perspective_db_engine

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tagger'])

CHUNK_SIZE = 100
NUM_WORKERS = 4


def count_ngrams(data):
    filename, spans = data

    text_file = TEXTS_DIR / filename
    text = text_file.read_text(encoding='utf-8', errors='replace')

    ngram_counts = []
    for begin, end in spans:
        x = text[begin:end].strip()
        ngram_counts.append(len(nlp(x)))
    return ngram_counts


def span_generator():
    df_chunks = pd.read_sql(
        'select R.filename, S.begin, S.end from responses as R, span_scores as S where R.filename = S.filename',
        con=perspective_db_engine(),
        chunksize=CHUNK_SIZE
    )

    for chunk in df_chunks:
        for filename, spans in chunk.groupby('filename'):
            spans = tuple(zip(spans.begin, spans.end))
            yield filename, spans


ngram_counts = []
g = span_generator()
with Pool(processes=NUM_WORKERS) as pool:
    for out in tqdm(pool.imap_unordered(count_ngrams, g, chunksize=CHUNK_SIZE)):
        ngram_counts.extend(out)

with open(OUTPUT_DIR / 'ngram_counts.pkl', 'wb') as f:
    pickle.dump(ngram_counts, f)
