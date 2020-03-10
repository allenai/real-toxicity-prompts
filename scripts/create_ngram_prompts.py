from sqlalchemy.orm import sessionmaker
import spacy
from tqdm import tqdm
import pickle

from utils.constants import DATA_DIR, TEXTS_DIR, PERSPECTIVE_API_KEY
from scripts.create_db import unpack_scores

from scripts.perspective_api_request import request
from utils.db import SpanScore, perspective_db_engine

Session = sessionmaker()
rps = 25


def create_ngrams_dataset():
    engine = perspective_db_engine()
    session = Session(bind=engine)

    nlp = spacy.load('en_core_web_sm')
    n = 5
    dataset = []

    span_score: SpanScore
    for i, span_score in enumerate(tqdm(session.query(SpanScore).filter(SpanScore.toxicity >= 0.9).all())):
        text_file = TEXTS_DIR / span_score.filename
        with text_file.open(errors='ignore') as f:
            f.seek(span_score.begin)
            span_text = f.read(span_score.end - span_score.begin).strip()

        doc = nlp(span_text)
        prompt, continuation = str(doc[:n]), str(doc[n:])
        dataset.append((span_score.filename, span_score.begin, span_score.end, prompt, continuation))

        if i == 50:
            break

    return dataset


def request_perspective_scores(data):
    filenames, begins, ends, prompts, continuations = zip(*data)

    prompt_responses = request(prompts, api_key=PERSPECTIVE_API_KEY, requests_per_second=rps)
    prompt_responses_unpacked = [unpack_scores(x) for x in prompt_responses if x]

    continuation_responses = request(continuations, api_key=PERSPECTIVE_API_KEY, requests_per_second=rps)
    continuation_responses_unpacked = [unpack_scores(x) if x else None for x in continuation_responses]

    return list(
        zip(filenames, begins, ends, prompts, prompt_responses_unpacked, continuations, continuation_responses_unpacked)
    )

def to_dataframe(data):
    filenames, begins, ends, prompts, prompt_responses_unpacked, continuations, continuation_responses_unpacked = zip(*data)
    {
        'filenames': filenames,

    }


pickle_prompts = DATA_DIR / 'ngram-beginning-prompts.pkl'
if pickle_prompts.exists():
    data = pickle.load(pickle_prompts.open('rb'))
else:
    data = create_ngrams_dataset()
    pickle.dump(data, pickle_prompts.open('wb'))

pickle_scores = DATA_DIR / 'ngram-beginning-prompts-with-scores.pkl'
scores = request_perspective_scores(data)
pickle.dump(scores, pickle_scores.open('wb'))
