import json
import logging
import math
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Iterable, List

import click
from knockknock import slack_sender
import pandas as pd
from tqdm.auto import tqdm

from scripts.perspective_api_request import perspective_api_request
from utils.generation import GPT2Generator
from utils.utils import batchify
from utils.constants import SLACK_CHANNEL, SLACK_WEBHOOK_URL

logging.disable(logging.CRITICAL)  # Disable logging from transformers


class PerspectiveWorker:
    SENTINEL = 'STOP'

    def __init__(self, out_file: Path, total: int, rps: int):
        self.requests_handled = set()
        for response in load_cache(out_file):
            self.requests_handled.add(response['request_id'])
        total -= len(self.requests_handled)

        # Setup worker thread
        self.task_queue = Queue()
        self.process = Process(target=self.perspective_worker, args=(self.task_queue, out_file, total, rps))
        self.process.start()

    def __call__(self, request_id: str, text: str):
        if request_id not in self.requests_handled:
            self.task_queue.put((request_id, text))

    def stop(self):
        self.task_queue.put(self.SENTINEL)
        self.process.join()

    @classmethod
    def perspective_worker(cls, queue: Queue, responses_file: Path, total: int, rps: int):
        queue_iter = iter(queue.get, cls.SENTINEL)
        pbar = tqdm(total=total, dynamic_ncols=True, position=1)
        perspective_api_request(queue_iter, responses_file=responses_file, pbar=pbar, requests_per_second=rps)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)


def generate(prompts: List[str],
             generator: GPT2Generator,
             max_len: int,
             batch_size: int,
             out_file: Path) -> Iterable[str]:
    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    for prompt in tqdm(batchify(prompts, batch_size),
                       total=math.ceil(len(prompts) / batch_size),
                       desc=f'Batch generation (bs={batch_size})',
                       dynamic_ncols=True):
        # Generate
        batch = generator.generate(prompt, max_len)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


@click.command()
@click.argument('out_dir')
@click.option('--dataset_file', required=True, type=str)
@click.option('--model_name_or_path', default='gpt2')
@click.option('--perspective_rps', default=25)
@click.option('--gen_samples', default=25)
@click.option('--gen_max_len', default=20)
@click.option('--gen_batch_size', default=32)
@click.option('--resume/--no-resume', default=False)
@slack_sender(webhook_url=SLACK_WEBHOOK_URL, channel=SLACK_CHANNEL)
def main(out_dir: str,
         dataset_file: str,
         model_name_or_path: str,
         perspective_rps: int,
         gen_samples: int,
         gen_max_len: int,
         gen_batch_size: int,
         resume: bool):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=resume)

    # Load dataset
    df = pd.read_csv(dataset_file)

    # Create perspective worker thread
    perspective = PerspectiveWorker(out_file=out_dir / 'perspective.jsonl',
                                    total=len(df) * gen_samples,
                                    rps=perspective_rps)

    # Generate and request perspective scores
    prompts = df['prompt.text'].repeat(gen_samples)
    generator = GPT2Generator(model_name_or_path)
    generations_iter = generate(prompts=prompts,
                                generator=generator,
                                max_len=gen_max_len,
                                batch_size=gen_batch_size,
                                out_file=out_dir / 'generations.jsonl')

    for i, gen in enumerate(generations_iter):
        perspective(f'generation-{i}', gen)

    perspective.stop()


if __name__ == '__main__':
    main()
