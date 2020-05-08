import json
import logging
import math
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict

import click
import pandas as pd
from tqdm.auto import tqdm

from scripts.perspective_api_request import perspective_api_request
from utils.generation import GPT2Generator
from utils.utils import batchify

logging.disable(logging.CRITICAL)  # Disable logging from transformers

SENTINEL = 'STOP'


class PerspectiveWorker:
    def __init__(self, out_file: Path, total: int, rps: int, requests_handled: Dict = None):
        self.requests_handled = requests_handled or {}

        # Setup worker thread
        self.task_queue = Queue()
        Process(target=self.perspective_worker, args=(self.task_queue, out_file, total, rps)).start()

    def request(self, request_id: str, text: str):
        if request_id not in self.requests_handled:
            self.task_queue.put((request_id, text))

    def stop(self):
        self.task_queue.put(SENTINEL)

    @staticmethod
    def perspective_worker(input: Queue, responses_file: Path, total: int, rps: int):
        pbar = tqdm(total=total, dynamic_ncols=True, position=1)
        perspective_api_request(iter(input.get, SENTINEL),
                                responses_file=responses_file,
                                pbar=pbar,
                                requests_per_second=rps)


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
                          model_path: str = None):
    # Store locations of output files
    config_file = out_dir / 'config.txt'
    perspective_file = out_dir / 'perspective.jsonl'
    generations_file = out_dir / 'generations.jsonl'
    dataset_file = out_dir / 'dataset.pkl'

    # Create config
    config = {
        'query': str(query),
        'model_path': model_path,
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
                perspective.request(f'prompt-{i}', prompt)

        if 'continuation_toxicity' not in df:
            for i, continuation in enumerate(df.continuation):
                perspective.request(f'continuation-{i}', continuation)

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
                        perspective.request(f'generation-{i}', generation)
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
                    perspective.request(f'generation-{i}', generation)
                i += 1

    if enable_perspective:
        perspective.stop()


@click.command()
@click.option('--model_name_or_path', required=True, type=str, help='model to use for generation')
@click.option('--out_dir', required=True, type=str, help='where to store outputs')
@click.option('--dataset_file', required=True, type=str)
@click.option('--gen_batch_size', required=True, type=int, help='batch size for generation')
@click.option('--perspective_rps', default=25, help='Perspective API rate limit (up to 25)')
@click.option('--num_gen_per_prompt', default=25)
@click.option('--max_gen_len', default=20)
def run_prompts_experiment(
        model_name_or_path: str,
        out_dir: str,
        gen_batch_size: int,
        perspective_rps: int,
        dataset_file: str,
        num_gen_per_prompt: int,
        max_gen_len: int,
):
    create_ngrams_dataset(
        df=pd.read_pickle(dataset_file),
        out_dir=Path(out_dir),
        generator=GPT2Generator(model_name_or_path),
        model_path=model_name_or_path,
        num_gen_per_prompt=num_gen_per_prompt,
        max_gen_len=max_gen_len,
        gen_batch_size=gen_batch_size,
        perspective_rps=perspective_rps
    )


if __name__ == '__main__':
    run_prompts_experiment()
