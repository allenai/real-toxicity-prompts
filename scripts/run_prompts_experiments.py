import json
import logging
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Dict

import click
import math
import pandas as pd
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel

from scripts.perspective_api_request import perspective_api_request
from utils.constants import OUTPUT_DIR
from utils.generation import GPT2Generator
from utils.utils import batchify

logging.disable(logging.CRITICAL)  # Disable logging from transformers

SENTINEL = 'STOP'


class PerspectiveWorker:
    def __init__(self, out_file: Path, total: int, rps: int, requests_handled: Dict = None):
        self.requests_handled = requests_handled or {}

        # Setup worker thread
        self.task_queue = Queue()
        self.done_queue = Queue()
        Process(
            target=self.perspective_worker,
            args=(self.task_queue, self.done_queue, out_file, total, rps)
        ).start()

    def add_request(self, request_id: str, text: str):
        if request_id not in self.requests_handled:
            self.task_queue.put((request_id, text))

    def get_responses(self) -> List[dict]:
        self.task_queue.put(SENTINEL)
        current_responses = self.done_queue.get()
        return list(self.requests_handled.values()) + current_responses

    @staticmethod
    def perspective_worker(input: Queue, output: Queue, responses_file: Path, total: int, rps: int):
        pbar = tqdm(total=total, dynamic_ncols=True, position=1)
        responses = perspective_api_request(iter(input.get, SENTINEL), responses_file=responses_file, pbar=pbar,
                                            requests_per_second=rps)
        output.put(responses)


def create_ngrams_dataset(df: pd.DataFrame,
                          out_dir: Path,
                          generator: GPT2Generator,
                          max_gen_len: int,
                          num_gen_per_prompt: int,
                          gen_batch_size: int,
                          perspective_rps: int,
                          enable_generation: bool = True,
                          enable_perspective: bool = True,
                          query: str = None,
                          model_path: Path = None) -> pd.DataFrame:
    # Store locations of output files
    config_file = out_dir / 'config.txt'
    perspective_file = out_dir / 'perspective.jsonl'
    generations_file = out_dir / 'generations.jsonl'
    dataset_file = out_dir / 'dataset.pkl'

    # Create config
    config = {
        'query': str(query),
        'model_path': str(model_path),
        'enable_generation': enable_generation,
        'enable_perspective_requests': enable_perspective,
        'max_gen_len': max_gen_len,
        'num_gen_per_prompt': num_gen_per_prompt,
        'model': repr(generator),
    }

    if out_dir.exists() and list(out_dir.iterdir()):
        # out_dir exists and has files
        if not config_file.exists():
            raise FileNotFoundError(f'Cannot resume experiment to output dir missing config file: {out_dir}')

        if json.load(config_file.open()) != config:
            raise FileExistsError(f'Config file does does not match current config: {config_file}')

        if dataset_file.exists():
            raise FileExistsError(f'Dataset already created: {dataset_file}')
    else:
        # Make directory if needed
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with config_file.open('w') as f:
            json.dump(config, f)

    print(f'Running experiment with outputs in {out_dir}...')
    print('Total rows:', len(df))

    perspective = None
    if enable_perspective:
        num_perspective_requests = 0
        if 'prompt_toxicity' not in df:
            print('Preparing prompts for perspective...')
            num_perspective_requests += len(df)
        if 'continuation_toxicity' not in df:
            print('Preparing continuations for perspective...')
            num_perspective_requests += len(df)
        if enable_generation and 'generation_toxicity' not in df:
            print(f'Preparing generations for perspective ({num_gen_per_prompt} generations per prompt)...')
            num_perspective_requests += len(df) * num_gen_per_prompt

        # Resume perspective
        requests_completed = {}
        if perspective_file.exists():
            with perspective_file.open() as f:
                for line in tqdm(f, desc='Loading saved perspective responses'):
                    # TODO: doesn't handle malformed lines
                    response = json.loads(line)
                    requests_completed[response['request_id']] = response
            print(f'Resuming perspective ({len(requests_completed)} already completed)...')
            num_perspective_requests -= len(requests_completed)

        # Create perspective worker thread
        perspective = PerspectiveWorker(perspective_file, num_perspective_requests, perspective_rps,
                                        requests_completed)

        if 'prompt_toxicity' not in df:
            for i, prompt in enumerate(df.prompt):
                perspective.add_request(f'prompt-{i}', prompt)

        if 'continuation_toxicity' not in df:
            for i, continuation in enumerate(df.continuation):
                perspective.add_request(f'continuation-{i}', continuation)

    # Generate
    if enable_generation and 'generations' not in df:
        i = 0
        generations = []

        # Resume generation
        if generations_file.exists():
            with generations_file.open() as f:
                for line in tqdm(f, desc='Loading saved generations'):
                    generation = json.loads(line)
                    generations.append(generation)
                    if enable_perspective:
                        perspective.add_request(f'generation-{i}', generation)
                    i += 1
            print(f'Resuming generation ({len(generations)} already computed)...')

        prompts = df.prompt.repeat(num_gen_per_prompt)[len(generations):]
        for prompt in tqdm(batchify(prompts, gen_batch_size),
                           total=math.ceil(len(prompts) / gen_batch_size),
                           desc=f'Generation (batch size {gen_batch_size})',
                           dynamic_ncols=True):
            # Generate
            batch = generator.generate(prompt, max_gen_len)

            for generation in batch:
                generations.append(generation)
                with generations_file.open('a') as f:
                    print(json.dumps(generation), file=f)
                if enable_perspective:
                    perspective.add_request(f'generation-{i}', generation)
                i += 1


@click.command()
@click.option('--model_name', required=True, type=str, help='model name in finetuned models dir')
@click.option('--gen_batch_size', required=True, type=int, help='batch size for generation (try 256)')
@click.option('--perspective_rps', required=True, type=int, help='Perspective API rate limit (up to 25)')
@click.option('--dataset_name', default='prompts_n_50percent')
@click.option('--num_gen_per_prompt', default=25)
@click.option('--max_gen_len', default=20)
def run_finetuned_experiment(
        model_name: str,
        gen_batch_size: int,
        perspective_rps: int,
        dataset_name: str,
        num_gen_per_prompt: int,
        max_gen_len: int,
):
    # Find directories
    prompts_dir = OUTPUT_DIR / 'prompts'
    experiments_dir = prompts_dir / 'experiments'
    finetune_dir = OUTPUT_DIR / 'finetuned_models'
    out_dir = experiments_dir / f'{dataset_name}_{model_name}'

    # Load dataset and model
    df = pd.read_pickle(prompts_dir / 'datasets' / f'{dataset_name}.pkl')
    model_path = finetune_dir / model_name / 'finetune_output'
    model = GPT2LMHeadModel.from_pretrained(model_path)
    generator = GPT2Generator(model)

    # Run experiment!
    create_ngrams_dataset(df=df,
                          out_dir=out_dir,
                          generator=generator,
                          model_path=model_path,
                          num_gen_per_prompt=num_gen_per_prompt,
                          max_gen_len=max_gen_len,
                          gen_batch_size=gen_batch_size,
                          perspective_rps=perspective_rps)


if __name__ == '__main__':
    run_finetuned_experiment()
