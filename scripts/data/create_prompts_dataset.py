from pathlib import Path
from typing import Union

import pandas as pd
import spacy
from spacy.tokens.doc import Doc
from sqlalchemy.sql.functions import random

from utils.constants import TEXTS_DIR, OPENWEBTEXT_DB
from utils.db import SpanScore, corpus_db_session
import click

# Span constants
MIN_SPAN_LEN = 64
MAX_SPAN_LEN = 1024
MAX_PROMPT_LEN = 128


def split_prompt(doc: Doc, n: int):
    if isinstance(n, float):
        # TODO: round n rather than flooring it
        n = round(n * len(doc))

    # Split text into prompt and continuation
    prompt = str(doc[:n])
    continuation = str(doc)[len(prompt):]  # Rather than taking remaining tokens, take the remainder of the string
    if len(prompt) == 0 or len(continuation) == 0 or len(prompt) > MAX_PROMPT_LEN:
        return None

    return prompt, continuation


def load_span_example(row: pd.Series, n: Union[int, float], nlp):
    # Load text from file
    text_file = TEXTS_DIR / row.filename
    try:
        text = text_file.read_text(encoding='utf-8', errors='strict')
    except UnicodeDecodeError:
        return None

    # Trim text
    text = text[row.begin:row.end].strip()
    if not (MIN_SPAN_LEN <= len(text) <= MAX_SPAN_LEN):
        return None

    # Tokenize text
    doc = nlp(text)
    return split_prompt(doc, n)


@click.command()
@click.option('--out_file', required=True, type=str)
@click.option('--n', required=True, type=float)
def create_prompts_dataset(out_file: str, n: float):
    out_file = Path(out_file)
    if out_file.exists():
        raise FileExistsError("Output file already exists.")
    if not OPENWEBTEXT_DB.exists():
        raise FileNotFoundError("Perspective database was not found.")

    session = corpus_db_session()

    # TODO: do this for all four ranges of toxicity
    query = (
        session.query(SpanScore)
            .filter(SpanScore.toxicity < 0.25)
            .filter(SpanScore.end - SpanScore.begin >= MIN_SPAN_LEN)
            .filter(SpanScore.end - SpanScore.begin <= MAX_SPAN_LEN)
            .order_by(random())
            .limit(30_000)
    )

    # Load dataframe from query and select relevant columns
    print("Reading from database...")
    df = pd.read_sql(query.statement, con=query.session.bind)
    df = df[['filename', 'begin', 'end', 'toxicity']]
    print(f"Returned {len(df)} rows")

    # Get prompts and continuations
    print("Loading text and tokenizing...")
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tagger'])
    examples = df.apply(lambda row: load_span_example(row, n, nlp), axis=1)

    # Add prompts and continuations to dataframe
    df = df[examples.notna()]
    df['prompt'], df['continuation'] = zip(*examples.dropna())
    print(f'Limited to {len(df)} rows after preprocessing')

    df = df.head(25_000)

    df.to_pickle(out_file)
    return df


if __name__ == '__main__':
    create_prompts_dataset()
