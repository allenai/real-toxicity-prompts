import json
import logging
import math
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Iterable, List

import click
import pandas as pd
import torch
import torch.nn.functional as F
from knockknock import slack_sender
from tqdm.auto import tqdm
from transformers.pipelines import pipeline

from models.affect_lm import AffectGPT2LMHeadModel
from scripts.perspective_api_request import perspective_api_request
from utils.constants import SLACK_CHANNEL, SLACK_WEBHOOK_URL
from utils.generation import GPT2Generator
from utils.pplm_generation import PPLMGeneration
from utils.utils import batchify

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
        print("Waiting for Perspective to finish...")
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


def pplm(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         out_file: Path):
    generator = PPLMGeneration('toxicity', device=0)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1
    assert num_cached_generations % num_samples == 0

    # Generate with prompts
    prompts = prompts[num_cached_generations // num_samples:]
    for prompt in tqdm(prompts, desc='Generation', dynamic_ncols=True):
        # Generate
        batch = generator(prompt,
                          length=max_len,
                          num_return_sequences=num_samples)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def ctrl(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         ctrl_code: str,
         model_name_or_path: str,
         out_file: Path) -> Iterable[str]:
    # Setup model
    generator = pipeline('text-generation', model=model_name_or_path, device=0)

    # Prepend CTRL code to prompts
    prompts = ctrl_code + " " + prompts
    print(prompts)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1
    assert num_cached_generations % num_samples == 0

    # Generate with prompts
    prompts = prompts[num_cached_generations // num_samples:]
    for prompt in tqdm(prompts, desc='Generation', dynamic_ncols=True):
        # Generate
        batch = generator(prompt,
                          num_return_sequences=num_samples,
                          do_sample=True,
                          temperature=1.0,
                          repetition_penalty=1.2,
                          top_k=0,
                          top_p=0.9,
                          max_length=max_len,
                          return_prompt=False)

        for generation in map(lambda g: g['generated_text'], batch):
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def _gpt2_helper(prompts: pd.Series,
                 max_len: int,
                 num_samples: int,
                 batch_size: int,
                 generator: GPT2Generator,
                 out_file: Path):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

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


def gpt2_ctrl(prompts: pd.Series,
              max_len: int,
              num_samples: int,
              batch_size: int,
              prompt_ctrl_code: str,
              ctrl_codes: List[str],
              model_name_or_path: str,
              out_file: Path):
    # Use default gpt2 architecture with additional tokens in vocab
    generator = GPT2Generator(model_name_or_path)
    num_tokens_added = generator.tokenizer.add_tokens(ctrl_codes)
    assert num_tokens_added == 2

    # Prepend ctrl code to prompts
    prompts = prompt_ctrl_code + prompts
    print(prompts)

    for generation in _gpt2_helper(prompts=prompts,
                                   max_len=max_len,
                                   num_samples=num_samples,
                                   batch_size=batch_size,
                                   generator=generator,
                                   out_file=out_file):
        yield generation


def gpt2_affect(prompts: pd.Series,
                max_len: int,
                num_samples: int,
                batch_size: int,
                target_class: int,
                num_classes: int,
                beta: int,
                model_name_or_path: str,
                out_file: Path) -> Iterable[str]:
    # Setup AffectGPT2 for generation
    model = AffectGPT2LMHeadModel.from_pretrained(model_name_or_path)
    generator = GPT2Generator(model)
    affect_label = F.one_hot(torch.LongTensor([target_class]), num_classes=num_classes).float().to(generator.device)
    model.set_affect_labels(affect_label)
    model.affect.beta = beta
    model.affect.ignore_special_tokens = True

    for generation in _gpt2_helper(prompts=prompts,
                                   max_len=max_len,
                                   num_samples=num_samples,
                                   batch_size=batch_size,
                                   generator=generator,
                                   out_file=out_file):
        yield generation


def gpt2(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         batch_size: int,
         model_name_or_path: str,
         out_file: Path) -> Iterable[str]:
    # Setup model
    generator = GPT2Generator(model_name_or_path)

    for generation in _gpt2_helper(prompts=prompts,
                                   max_len=max_len,
                                   num_samples=num_samples,
                                   batch_size=batch_size,
                                   generator=generator,
                                   out_file=out_file):
        yield generation


@click.command()
@click.argument('out_dir')
@click.option('--dataset_file', required=True, type=str)
@click.option('--model_type', required=True, type=click.Choice(['gpt2', 'ctrl', 'gpt2-affect', 'gpt2-ctrl', 'pplm']))
@click.option('--model_name_or_path', default='gpt2')
@click.option('--perspective_rps', default=25)
@click.option('--gen_samples', default=25)
@click.option('--gen_max_len', default=20)
@click.option('--gen_batch_size', default=32)
@click.option('--resume/--no-resume', default=False)
@slack_sender(webhook_url=SLACK_WEBHOOK_URL, channel=SLACK_CHANNEL)
def main(out_dir: str,
         dataset_file: str,
         model_type: str,
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
    prompts = df['prompt.text']

    # Create perspective worker thread
    perspective = PerspectiveWorker(out_file=out_dir / 'perspective.jsonl',
                                    total=len(df) * gen_samples,
                                    rps=perspective_rps)

    # Generate and request perspective scores
    if model_type == 'gpt2':
        generations_iter = gpt2(prompts=prompts,
                                max_len=gen_max_len,
                                num_samples=gen_samples,
                                batch_size=gen_batch_size,
                                model_name_or_path=model_name_or_path,
                                out_file=out_dir / 'generations.jsonl')
    elif model_type == 'gpt2-affect':
        generations_iter = gpt2_affect(prompts=prompts,
                                       max_len=gen_max_len,
                                       num_samples=gen_samples,
                                       batch_size=gen_batch_size,
                                       model_name_or_path=model_name_or_path,
                                       out_file=out_dir / 'generations.jsonl',
                                       # Affect
                                       target_class=0,
                                       num_classes=2,
                                       beta=3)
    elif model_type == 'gpt2-ctrl':
        generations_iter = gpt2_ctrl(prompts=prompts,
                                     max_len=gen_max_len,
                                     num_samples=gen_samples,
                                     batch_size=gen_batch_size,
                                     model_name_or_path=model_name_or_path,
                                     out_file=out_dir / 'generations.jsonl',
                                     # GPT2-CTRL
                                     prompt_ctrl_code='<|nontoxic|>',
                                     ctrl_codes=['<|nontoxic|>', '<|toxic|>'])
    elif model_type == 'ctrl':
        generations_iter = ctrl(prompts=prompts,
                                max_len=gen_max_len,
                                num_samples=gen_samples,
                                model_name_or_path=model_name_or_path,
                                out_file=out_dir / 'generations.jsonl',
                                # CTRL
                                ctrl_code='Links')
    elif model_type == 'pplm':
        generations_iter = pplm(prompts=prompts,
                                max_len=gen_max_len,
                                num_samples=gen_samples,
                                out_file=out_dir / 'generations.jsonl')
    else:
        raise NotImplementedError(f'Model {model_name_or_path} not implemented')

    for i, gen in enumerate(generations_iter):
        perspective(f'generation-{i}', gen)

    torch.cuda.empty_cache()

    perspective.stop()
    print('Done')


if __name__ == '__main__':
    main()
