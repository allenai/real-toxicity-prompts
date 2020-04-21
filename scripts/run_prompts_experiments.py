import json
import logging
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Dict

import pandas as pd
import spacy
import torch
from spacy.tokens.doc import Doc
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel

from datasets.affect_dataset import create_affect_vector
from models.affect_lm import AffectGPT2LMHeadModel
from scripts.create_db import unpack_scores
from scripts.perspective_api_request import perspective_api_request
from utils.constants import TEXTS_DIR, OUTPUT_DIR, PERSPECTIVE_DB
from utils.db import SpanScore, perspective_db_session
from utils.generation import GPT2Generator
from utils.utils import batchify

SENTINEL = 'STOP'
logging.disable(logging.CRITICAL)  # Disable logging from transformers

# Span constants
MIN_SPAN_LEN = 64
MAX_SPAN_LEN = 1024
MAX_PROMPT_LEN = 128

# Generation hyperparameters
GENERATION_BATCH_SIZE = 25


class PerspectiveWorker:
    def __init__(self, out_file: Path, total: int, requests_handled=None):
        self.requests_handled = requests_handled or set()

        # Setup worker thread
        self.task_queue = Queue()
        self.done_queue = Queue()
        Process(
            target=self.perspective_worker,
            args=(self.task_queue, self.done_queue, out_file, total)
        ).start()

    def add_request(self, request_id: str, text: str):
        if request_id not in self.requests_handled:
            self.task_queue.put((request_id, text))

    def get_responses(self) -> List[dict]:
        self.task_queue.put(SENTINEL)
        return self.done_queue.get()

    @staticmethod
    def perspective_worker(input: Queue, output: Queue, responses_file: Path, total: int):
        pbar = tqdm(total=total, dynamic_ncols=True)
        responses = perspective_api_request(iter(input.get, SENTINEL), responses_file=responses_file, pbar=pbar)
        output.put(responses)


def extract_toxicity_scores(response_dicts: List[dict]) -> Dict[str, List[float]]:
    scores = {}
    for response_dict in response_dicts:
        response_col = response_dict['request_id'].split('-')[0]
        response = response_dict['response']
        if response:
            summary_scores, span_scores = unpack_scores(response)
            toxicity = summary_scores['toxicity']
        else:
            toxicity = None
        scores.setdefault(response_col, []).append(toxicity)
    return scores


def split_prompt(doc: Doc, n: int):
    if isinstance(n, float):
        # TODO: round n rather than flooring it
        n = int(n * len(doc))

    # Split text into prompt and continuation
    prompt = str(doc[:n])
    continuation = str(doc)[len(prompt):]  # Rather than taking remaining tokens, take the remainder of the string
    if len(prompt) == 0 or len(continuation) == 0 or len(prompt) > MAX_PROMPT_LEN:
        return None

    return prompt, continuation


def preprocess_data(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if 'prompt' in df and 'continuation' in df:
        return df

    # Split data
    print(f"Splitting spans into (prompt, continuation) pairs with n={n}...")
    df = df[df.examples.notna()]
    split_spans = df.examples.apply(lambda x: split_prompt(x, n))
    df = df[split_spans.notna()]
    df['prompt'], df['continuation'] = zip(*split_spans.dropna())
    df.drop(columns='examples')
    print(f'Limited to {len(df)} rows after preprocessing')
    return df


def generate_with_prompt(prompt: str,
                         generator: GPT2Generator,
                         max_len: int,
                         num_return_sequences: int) -> List[str]:
    generations = []
    for i in range(0, num_return_sequences, GENERATION_BATCH_SIZE):
        batch_gen = generator.generate_multiple(
            prompt,
            max_len=max_len,
            num_return_sequences=min(num_return_sequences - i, GENERATION_BATCH_SIZE)
        )
        generations.extend(batch_gen)
    return generations


def create_ngrams_dataset(df: pd.DataFrame,
                          out_dir: Path,
                          generator: GPT2Generator,
                          max_gen_len: int = 20,
                          num_gen_per_prompt: int = 25,
                          enable_generation: bool = True,
                          enable_perspective: bool = True) -> pd.DataFrame:
    # Store locations of output files
    config_file = out_dir / 'config.txt'
    perspective_file = out_dir / 'perspective.jsonl'
    generations_file = out_dir / 'generations.jsonl'
    dataset_file = out_dir / 'dataset.pkl'

    # Create config
    config = {
        'enable_generation': enable_generation,
        'enable_perspective_requests': enable_perspective,
        'max_gen_len': max_gen_len,
        'num_gen_per_prompt': num_gen_per_prompt,
        'model': repr(generator),
    }

    if out_dir.exists():
        if config_file.exists() and json.load(config_file.open()) != config:
            raise FileExistsError(f'Config file already exists and does not match current config: {config_file}')

        if dataset_file.exists():
            raise FileExistsError(f'Dataset already created: {dataset_file}')
    else:
        # Make directory
        out_dir.mkdir(parents=True)

        # Save config
        with config_file.open('w') as f:
            json.dump(config, f)

    print(f'Running experiment with outputs in {out_dir}...')

    perspective = None
    if enable_perspective:
        # Resume perspective
        requests_completed = None
        if perspective_file.exists():
            requests_completed = set()
            with perspective_file.open() as f:
                for line in f:
                    # TODO: doesn't handle malformed lines
                    response = json.loads(line)
                    request_id = response['request_id']
                    requests_completed.add(request_id)
            print(f'Resuming perspective ({len(requests_completed)} already completed)...')

        num_perspective_requests = sum([
            len(df) if 'prompt_toxicity' not in df else 0,
            len(df) if 'continuation_toxicity' not in df else 0,
            len(df) * num_gen_per_prompt if enable_generation and 'generation_toxicity' not in df else 0
        ])
        perspective = PerspectiveWorker(perspective_file, num_perspective_requests, requests_completed)

        if 'prompt_toxicity' not in df:
            for i, prompt in enumerate(df.prompt):
                perspective.add_request(f'prompt-{i}', prompt)

        if 'continuation_toxicity' not in df:
            for i, continuation in enumerate(df.continuation):
                perspective.add_request(f'continuation-{i}', continuation)

    # Generate
    if enable_generation and 'generations' not in df:
        generations = []

        # Resume generation
        if generations_file.exists():
            with generations_file.open() as f:
                for line in f:
                    # TODO: doesn't handle malformed lines
                    generations.append(json.loads(line))
            print(f'Resuming generation ({len(generations)} already computed)...')

        num_generations = len(df.prompt) - len(generations)
        for batch_i, prompt in tqdm(enumerate(df.prompt), total=num_generations, desc='Generation', dynamic_ncols=True,
                                    position=1):
            if batch_i < len(generations):
                # Handle generations already computed
                batch = generations[batch_i]
            else:
                # Generate
                batch = generate_with_prompt(prompt, generator, max_gen_len, num_gen_per_prompt)
                generations.append(batch)
                with generations_file.open('a') as f:
                    print(json.dumps(batch), file=f)

            # Request perspective scores
            if enable_perspective:
                for i, generation in enumerate(batch):
                    perspective.add_request(f'generation-{batch_i},{i}', generation)

        df['generation'] = generations

    # Extract toxicity scores from perspective responses
    if enable_perspective:
        response_dicts = perspective.get_responses()

        for request_column, scores in extract_toxicity_scores(response_dicts).items():
            score_column = f'{request_column}_toxicity'
            assert score_column not in df
            if request_column == 'generation':
                scores = list(batchify(scores, num_gen_per_prompt))
            df[score_column] = scores

    # Save data
    df.to_pickle(dataset_file)
    return df


def load_span_example(row: pd.Series, nlp):
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
    return doc


def load_features(cached_features_file: Path):
    if cached_features_file.exists():
        print("Loading cached features...")
        df = pd.read_pickle(cached_features_file)
    else:
        session = perspective_db_session()
        query = (
            session.query(SpanScore)
                .filter(SpanScore.toxicity >= .75)
                .filter(SpanScore.end - SpanScore.begin >= MIN_SPAN_LEN)
                .filter(SpanScore.end - SpanScore.begin <= MAX_SPAN_LEN)
        )

        if not PERSPECTIVE_DB.exists():
            raise FileNotFoundError("Perspective database was not found. Try using a cached features file.")

        # Load dataframe from query and select relevant columns
        print("Reading from database...")
        df = pd.read_sql(query.statement, con=query.session.bind)
        df = df[['filename', 'begin', 'end', 'toxicity']]
        print(f"Returned {len(df)} rows")

        # Get prompts and continuations
        print("Loading text and tokenizing...")
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tagger'])
        df['examples'] = df.apply(lambda row: load_span_example(row, nlp), axis=1)

        df.to_pickle(cached_features_file)

    return df


# TODO: use this for affect generation
def load_affect_generator(model_path: Path):
    # Use AffectGPT2 for generation
    affect_lm = AffectGPT2LMHeadModel.from_pretrained(model_path)
    generator = GPT2Generator(affect_lm)

    # Create non-toxic affect vector
    affect_labels = torch.tensor(create_affect_vector(non_toxic=1.5), device=generator.device)
    affect_lm.set_affect_labels(affect_labels)

    # Set beta to 1
    affect_lm.affect.beta = 1

    return generator


def run_experiments():
    prompts_dir = OUTPUT_DIR / 'prompts'
    experiments_dir = prompts_dir / 'experiments'
    finetune_dir = OUTPUT_DIR / 'finetune'

    n_50_percent_df = pd.read_pickle(prompts_dir / 'datasets' / 'prompts_n_50percent.pkl')

    experiment_kwargs = [
        {
            'df': n_50_percent_df,
            'model_path': finetune_dir / 'finetune_toxicity_percentile_lte2' / 'finetune_output',
            'out_dir': experiments_dir / 'prompts_n_50percent_finetune_toxicity_percentile_lte2',
            'num_gen_per_prompt': 25
        },
        {
            'df': n_50_percent_df,
            'model_path': finetune_dir / 'finetune_toxicity_percentile_middle_20_subsample' / 'finetune_output',
            'out_dir': experiments_dir / 'prompts_n_50percent_finetune_toxicity_percentile_middle_20_subsample',
            'num_gen_per_prompt': 25
        },
        {
            'df': n_50_percent_df,
            'model_path': finetune_dir / 'finetune_toxicity_percentile_gte99' / 'finetune_output',
            'out_dir': experiments_dir / 'prompts_n_50percent_finetune_toxicity_percentile_gte99',
            'num_gen_per_prompt': 25
        },
    ]

    for kwargs in experiment_kwargs:
        model = GPT2LMHeadModel.from_pretrained(kwargs['model_path'])
        generator = GPT2Generator(model)
        try:
            create_ngrams_dataset(generator=generator, **kwargs)
        except FileExistsError as e:
            print(e)
            print('Skipping experiment...')


if __name__ == '__main__':
    run_experiments()
