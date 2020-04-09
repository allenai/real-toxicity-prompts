import logging
import pickle
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.functions import random
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from utils.constants import TEXTS_DIR, PERSPECTIVE_API_ATTRIBUTES
from utils.db import perspective_db_session, SpanScore

Session = sessionmaker()
logger = logging.getLogger(__name__)

NUM_AFFECTS = len(PERSPECTIVE_API_ATTRIBUTES)


def create_affect_vector(toxicity: float = 0.,
                         severe_toxicity: float = 0.,
                         identity_attack: float = 0.,
                         insult: float = 0.,
                         threat: float = 0.,
                         profanity: float = 0.,
                         sexually_explicit: float = 0.,
                         flirtation: float = 0.) -> List[float]:
    return [
        insult,
        severe_toxicity,
        toxicity,
        profanity,
        sexually_explicit,
        flirtation,
        identity_attack,
        threat
    ]


class AffectDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 args,
                 parent_dir: Path,
                 evaluate: bool,
                 block_size=512,
                 test_size=0.1):
        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        cached_features_file = (
                parent_dir / f'{args.model_type}_cached_lm_{args.block_size}_affect.txt'
        )

        if cached_features_file.exists() and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with cached_features_file.open('rb') as handle:
                cached = pickle.load(handle)
        else:
            df = self.load_perspective_rows()

            # Split data and save with metadata
            train, test = train_test_split(df, test_size=test_size)
            train.to_pickle(parent_dir / 'train.pkl')
            test.to_pickle(parent_dir / 'test.pkl')

            logger.info(f"Creating features from perspective database query")
            train_examples = self.create_examples(train, tokenizer, block_size)
            test_examples = self.create_examples(test, tokenizer, block_size)

            logger.info("Saving features into cached file %s", cached_features_file)
            cached = {'train': train_examples, 'test': test_examples}
            with cached_features_file.open('wb') as handle:
                pickle.dump(cached, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Load examples
        self.examples = cached['test'] if evaluate else cached['train']

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        input_ids, affect = self.examples[item]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(affect, dtype=torch.float)

    @staticmethod
    def create_examples(df: pd.DataFrame, tokenizer: PreTrainedTokenizer, block_size: int):
        examples = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='Creating examples'):
            # Load text and tokenize
            text_file = TEXTS_DIR / row.filename
            text = text_file.read_text(encoding='utf-8', errors='replace')[row.begin:row.end].strip()
            tokens = tokenizer.encode(text, max_length=block_size)

            # Create affect vector from row
            affect = create_affect_vector(
                row.toxicity,
                row.severe_toxicity,
                row.identity_attack,
                row.insult,
                row.threat,
                row.profanity,
                row.sexually_explicit,
                row.flirtation
            )
            affect = np.array(affect).round().astype(int).tolist()

            examples.append((tokens, affect))

        return examples

    @staticmethod
    def load_perspective_rows(row_limit=100_000) -> pd.DataFrame:
        logger.info(f"Querying {row_limit} rows from perspective database")

        session = perspective_db_session()
        query = session.query(SpanScore).order_by(random()).limit(row_limit)
        df = pd.read_sql(query.statement, con=query.session.bind)

        # toxicity_query = session.query(SpanScore).order_by(SpanScore.toxicity.desc()).limit(5_000)
        # toxic_df = pd.read_sql(toxicity_query.statement, con=toxicity_query.session.bind)
        #
        # # Append sampled toxic rows
        # # TODO: remove duplicates
        # df = df.append(toxic_df).drop_duplicates(subset=['filename', 'begin', 'end'])

        if len(df) < row_limit:
            raise RuntimeError("Selected perspective subset not large enough to subsample from")

        return df
