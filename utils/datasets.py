import logging
import pickle
from pathlib import Path
from typing import List

import torch
from sqlalchemy.orm import sessionmaker
from torch.utils.data import Dataset
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
                         flirtation: float = 0.) -> torch.Tensor:
    return torch.tensor([
        insult,
        severe_toxicity,
        toxicity,
        profanity,
        sexually_explicit,
        flirtation,
        identity_attack,
        threat
    ])


class AffectDataset(Dataset):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 args,
                 cached_features_file: Path,
                 block_size=512):
        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        if cached_features_file.exists() and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with cached_features_file.open('rb') as handle:
                self.examples = pickle.load(handle)
        else:
            rows = self.load_perspective_rows()

            logger.info(f"Creating features from perspective database query")

            self.examples = []
            for row in rows:
                # Load text and tokenize
                text_file = TEXTS_DIR / row.filename
                text = text_file.read_text(encoding='utf-8')
                tokens = tokenizer.encode(text, max_length=block_size, pad_to_max_length=True, return_tensors='pt')

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
                ).round()

                self.examples.append((tokenizer.build_inputs_with_special_tokens(tokens), affect))

                logger.info("Saving features into cached file %s", cached_features_file)
                with cached_features_file.open('wb') as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def load_perspective_rows(limit_rows=100_000) -> List[SpanScore]:
        logger.info(f"Querying {limit_rows} rows from perspective database")

        session = perspective_db_session()

        low_tox_rows = session.query(SpanScore). \
            order_by(SpanScore.toxicity). \
            limit(limit_rows // 2). \
            all()
        high_tox_rows = session.query(SpanScore). \
            order_by(SpanScore.toxicity.desc()). \
            limit(limit_rows // 2). \
            all()

        if len(low_tox_rows) + len(high_tox_rows) < limit_rows:
            raise RuntimeError("Selected perspective subset not large enough to subsample from")

        rows = low_tox_rows + high_tox_rows
        return rows
