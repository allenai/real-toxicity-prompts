import pickle

import pandas as pd
import spacy
from sqlalchemy.orm import sessionmaker
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm

from scripts.create_db import unpack_scores
from scripts.perspective_api_request import request
from utils.constants import DATA_DIR, TEXTS_DIR, PERSPECTIVE_API_KEY
from utils.db import SpanScore, perspective_db_engine

Session = sessionmaker()
rps = 25


class NgramPromptsDataset(Dataset):
    def __init__(self, n, rows, nlp):
        super().__init__()
        self.n = n
        self.rows = rows
        self.nlp = nlp

    def __getitem__(self, idx):
        span_score = self.rows[idx]

        text_file = TEXTS_DIR / span_score.filename
        with open(text_file, errors='ignore') as f:
            f.seek(span_score.begin)
            text = f.read(span_score.end - span_score.begin).strip()

        doc = self.nlp(text)
        prompt, continuation = str(doc[:self.n]), str(doc[self.n:])
        return span_score.filename, span_score.begin, span_score.end, prompt, continuation

    def __len__(self):
        return len(self.rows)


def create_ngrams_dataset():
    engine = perspective_db_engine()
    session = Session(bind=engine)

    nlp = spacy.load('en_core_web_sm')
    n = 5
    batch_size = 64
    num_workers = 8

    rows = session.query(SpanScore).filter(SpanScore.toxicity >= 0.75).all()
    dataset = NgramPromptsDataset(n, rows, nlp)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, sampler=sampler, num_workers=num_workers,
                            collate_fn=lambda x: x)

    dataset = []
    for i, batch in enumerate(tqdm(dataloader)):
        dataset.extend(batch)

    return dataset


def request_perspective_scores(data):
    filenames, begins, ends, prompts, continuations = zip(*data)

    prompt_responses = request(prompts, api_key=PERSPECTIVE_API_KEY, requests_per_second=rps)
    prompt_responses_unpacked = [unpack_scores(x) if x else None for x in prompt_responses]

    continuation_responses = request(continuations, api_key=PERSPECTIVE_API_KEY, requests_per_second=rps)
    continuation_responses_unpacked = [unpack_scores(x) if x else None for x in continuation_responses]

    return list(
        zip(filenames, begins, ends, prompts, prompt_responses_unpacked, continuations, continuation_responses_unpacked)
    )


def to_dataframe(data):
    filenames, begins, ends, prompts, prompt_responses_unpacked, continuations, continuation_responses_unpacked = zip(
        *data)
    d = {
        'filename': filenames,
        'begin': begins,
        'end': ends,
        'prompt': prompts,
        'continuations': continuations,
        'prompt_toxicity': [x[0]['toxicity'] if x else None for x in prompt_responses_unpacked],
        'continuation_toxicity': [x[0]['toxicity'] if x else None for x in continuation_responses_unpacked]
    }
    return pd.DataFrame(d)


def create_ngram_prompts():
    data = create_ngrams_dataset()
    scores = request_perspective_scores(data)
    df = to_dataframe(scores)
    return df


df = create_ngram_prompts()
pkl = DATA_DIR / 'ngram-beginning-prompts.pkl'
pickle.dump(df, pkl.open('wb'))
