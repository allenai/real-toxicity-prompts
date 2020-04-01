import json
import shutil
from math import ceil
from multiprocessing.pool import Pool
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import spacy
from sqlalchemy.orm import Query
from tqdm.auto import tqdm

from scripts.create_db import unpack_scores
from scripts.perspective_api_request import request
from utils.constants import TEXTS_DIR, OUTPUT_DIR
from utils.db import SpanScore, perspective_db_session
from utils.generation import GPT2Generator

nlp = spacy.load('en_core_web_sm')


def load_span_example(row, n: Union[int, float], max_prompt_len=128) -> Optional[Tuple[str, str]]:
    text_file = TEXTS_DIR / row.filename
    try:
        text = text_file.read_text(encoding='utf-8', errors='strict')
    except UnicodeDecodeError:
        return None

    text = text[row.begin:row.end].strip()
    doc = nlp(text)

    if isinstance(n, float):
        n = int(n * len(doc))
    prompt, continuation = str(doc[:n]), str(doc[n:])

    if len(prompt) == 0 or len(continuation) == 0 or len(prompt) > max_prompt_len:
        return None

    return prompt, continuation


def generate(corpus: List[str], generator, max_len, batch_size=8) -> List[str]:
    batches = np.array_split(corpus, ceil(len(corpus) / batch_size))
    generations = []
    for batch in tqdm(batches, desc='generating', position=1):
        generations.extend(generator.generate(batch, max_len=max_len))
    return generations


def perspective_request(corpus: List[str], file: Optional[Path]) -> List[dict]:
    prompt_responses = request(corpus, responses_file=file)
    toxicity_scores = []
    for r in prompt_responses:
        if not r:
            toxicity_scores.append(None)
            continue
        summary_scores, span_scores = unpack_scores(r)
        toxicity = summary_scores['toxicity']
        toxicity_scores.append(toxicity)
    return toxicity_scores


def create_ngrams_dataset(query: Query,
                          n: Union[int, float],
                          should_score_generations: bool = False,
                          out_dir: Optional[Path] = None,
                          generator: GPT2Generator = GPT2Generator(),
                          max_generation_len=50) -> pd.DataFrame:
    # Make directory
    if out_dir:
        out_dir.mkdir(parents=True)

        # Save config
        config_file = out_dir / 'config.txt'
        config = {
            'query': str(query.statement),
            'n': n,
            'generation_len': max_generation_len,
            'generator': repr(generator)
        }
        with config_file.open('w') as f:
            json.dump(config, f)

    # Load dataframe from query and select relevant columns
    print("reading from database...")
    df = pd.read_sql(query.statement, con=query.session.bind)
    df = df[['filename', 'begin', 'end', 'toxicity']]
    print(f"returned {len(df)} rows")

    # Get prompts and continuations
    examples = df.apply(lambda row: load_span_example(row, n), axis=1)
    df = df[examples.notna()]
    df['prompt'], df['continuation'] = zip(*examples.dropna())
    df = df.reset_index(drop=True)
    print(f'Limited to {len(df)} rows after preprocessing')

    # Get generations and perspective scores
    with Pool(processes=1) as pool:
        prompts, continuations = df.prompt.tolist(), df.continuation.tolist()

        # Get perspective responses in background
        perspective_results = pool.starmap_async(
            perspective_request,
            [
                (prompts, out_dir / 'prompts.jsonl' if out_dir else None),
                (continuations, out_dir / 'continuations.jsonl' if out_dir else None)
            ]
        )
        generations = generate(prompts, generator, max_generation_len)
        df['generation'] = generations
        df['prompt_toxicity'], df['continuation_toxicity'] = perspective_results.get()

    if should_score_generations:
        df['generation_toxicity'] = perspective_request(
            generations,
            out_dir / 'generations.jsonl' if out_dir else None
        )

    if out_dir:
        df.to_pickle(out_dir / 'dataset.pkl')
    return df


def test_create_ngrams_dataset():
    out_dir = OUTPUT_DIR / 'prompts' / 'test-prompts'
    if out_dir.exists():
        shutil.rmtree(out_dir)

    session = perspective_db_session()
    query = (
        session.query(SpanScore)
            .filter(SpanScore.toxicity >= .75)
            .filter(SpanScore.end - SpanScore.begin >= 128)
            .filter(SpanScore.end - SpanScore.begin <= 1024)
            .limit(1000)
    )

    df = create_ngrams_dataset(query, n=.2, should_score_generations=False, out_dir=out_dir)
    print(df)


def run_experiments():
    session = perspective_db_session()
    out_dir = OUTPUT_DIR / 'prompts'

    experiment_kwargs = [
        {
            'query': session.query(SpanScore)
                .filter(SpanScore.toxicity >= .75)
                .filter(SpanScore.end - SpanScore.begin >= 64)
                .filter(SpanScore.end - SpanScore.begin <= 1024)
            ,
            'n': 0.2,
            'should_score_generations': True,
            'out_dir': out_dir / 'prompts_n_20percent'
        },
        {
            'query': session.query(SpanScore)
                .filter(SpanScore.toxicity >= .75)
                .filter(SpanScore.end - SpanScore.begin >= 64)
                .filter(SpanScore.end - SpanScore.begin <= 1024)
            ,
            'n': 5,
            'should_score_generations': True,
            'out_dir': out_dir / 'prompts_n_5'
        },
        {
            'query': session.query(SpanScore)
                .filter(SpanScore.toxicity >= .75)
                .filter(SpanScore.end - SpanScore.begin >= 64)
                .filter(SpanScore.end - SpanScore.begin <= 1024)
            ,
            'n': 10,
            'should_score_generations': True,
            'out_dir': out_dir / 'prompts_n_10'
        },
        {
            'query': session.query(SpanScore)
                .filter(SpanScore.toxicity >= .75)
                .filter(SpanScore.end - SpanScore.begin >= 64)
                .filter(SpanScore.end - SpanScore.begin <= 1024)
            ,
            'n': 15,
            'should_score_generations': True,
            'out_dir': out_dir / 'prompts_n_15'
        }
    ]

    generator = GPT2Generator()

    for kwargs in experiment_kwargs:
        create_ngrams_dataset(generator=generator, **kwargs)


if __name__ == '__main__':
    run_experiments()
