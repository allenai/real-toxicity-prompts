import json
import logging
import math
from functools import partial
from multiprocessing import Process, Queue
from pathlib import Path
import pickle
from typing import Iterable, Optional

import click
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers.pipelines import pipeline

from models.affect_lm import AffectGPT2LMHeadModel
from utils.perspective_api import perspective_api_request
from utils.generation import GPT2Generator
from utils.pplm_generation import PPLMGeneration
from utils.utils import batchify
from utils.xlm_generation import XLNetGenerator

logging.disable(logging.CRITICAL)  # Disable logging from transformers


class PerspectiveWorker:
    SENTINEL = 'STOP'

    def __init__(self, out_file: Path, total: int, rps: int):
        if not rps:
            print("Disabling perspective (rps is 0)")
            self.enabled = False
            return

        self.enabled = True

        self.requests_handled = set()
        for response in load_cache(out_file):
            self.requests_handled.add(response['request_id'])
        total -= len(self.requests_handled)

        # Setup worker thread
        self.task_queue = Queue()
        self.process = Process(target=self.perspective_worker, args=(self.task_queue, out_file, total, rps))
        self.process.start()

    def __call__(self, request_id: str, text: str):
        if not self.enabled:
            return

        if request_id not in self.requests_handled:
            self.task_queue.put((request_id, text))

    def stop(self):
        if not self.enabled:
            return

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


def xlm(prompts: pd.Series,
        max_len: int,
        num_samples: int,
        batch_size: int,
        out_file: Path,
        **generate_kwargs):
    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1
    prompts = prompts[num_cached_generations:]

    if prompts.empty:
        return

    # Setup model
    generator = XLNetGenerator()
    print("Loaded XLNetGenerator")

    # Generate
    for prompt in tqdm(batchify(prompts, batch_size),
                       total=math.ceil(len(prompts) / batch_size),
                       desc=f'Batch generation (bs={batch_size})',
                       dynamic_ncols=True):
        # Generate
        try:
            batch = generator(prompt, max_len=max_len, **generate_kwargs)
        except RuntimeError as e:
            print("Error during generation with prompt:", prompt)
            print(e)
            print("Emptying CUDA cache and retrying...")
            torch.cuda.empty_cache()

            batch = ["GENERATION_ERROR_CUDA"] * len(prompt)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def pplm(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         batch_size: int,
         class_label: int,
         num_iterations: int,
         model_name_or_path: str,
         out_file: Path):
    # Set up PPLM with multiprocessing
    generator = PPLMGeneration(model_name_or_path, device=0)
    ctx = mp.get_context('spawn')
    generator.model.share_memory()
    generator.classifier.share_memory()
    pplm_func = partial(generator.__call__, class_label=class_label, num_iterations=num_iterations, length=max_len)

    # Repeat prompts
    prompts = prompts.repeat(num_samples)

    # Resume generation
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Generate with prompts
    prompts = prompts[num_cached_generations:]
    with ctx.Pool(processes=batch_size) as pool:
        for batch in tqdm(pool.imap(pplm_func, prompts), total=len(prompts), desc='Generation', dynamic_ncols=True):
            for generation in batch:
                with out_file.open('a') as f:
                    print(json.dumps(generation), file=f)
                yield generation


def _pipeline_helper(prompts: pd.Series,
                     model_name_or_path: str,
                     max_len: int,
                     num_samples: int,
                     out_file: Path,
                     **generate_kwargs):
    # Load cached generations
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1
    assert num_cached_generations % num_samples == 0

    # Remove prompts that have already been generated with
    prompts = prompts[num_cached_generations // num_samples:]
    if prompts.empty:
        return

    # Setup model
    generator = pipeline('text-generation', model=model_name_or_path, device=0)
    print("Created pipeline with model:", generator.model.__class__.__name__)

    # Generate with prompts
    for prompt in tqdm(prompts, desc='Generation', dynamic_ncols=True):
        # Generate
        try:
            batch = generator(prompt,
                              num_return_sequences=num_samples,
                              clean_up_tokenization_spaces=True,
                              do_sample=True,
                              top_k=0,
                              top_p=0.9,
                              max_length=max_len,
                              return_prompt=False,
                              **generate_kwargs)
            batch = map(lambda g: g['generated_text'], batch)
        except RuntimeError as e:
            print("Error during generation with prompt:", prompt)
            print(e)
            print("Emptying CUDA cache and retrying...")
            torch.cuda.empty_cache()

            batch = ["GENERATION_ERROR_CUDA"] * num_samples

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def openai_gpt(prompts: pd.Series,
               max_len: int,
               num_samples: int,
               model_name_or_path: str,
               out_file: Path,
               **generate_kwargs):
    yield from _pipeline_helper(prompts=prompts,
                                model_name_or_path=model_name_or_path,
                                max_len=max_len,
                                num_samples=num_samples,
                                out_file=out_file,
                                **generate_kwargs)


def ctrl(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         ctrl_code: str,
         model_name_or_path: str,
         out_file: Path,
         **generate_kwargs) -> Iterable[str]:
    # Prepend CTRL code to prompts
    prompts = ctrl_code + " " + prompts
    print(prompts)

    yield from _pipeline_helper(prompts=prompts,
                                model_name_or_path=model_name_or_path,
                                max_len=max_len,
                                num_samples=num_samples,
                                out_file=out_file,
                                **generate_kwargs)


def _gpt2_helper(prompts: pd.Series,
                 max_len: int,
                 num_samples: int,
                 batch_size: int,
                 generator: GPT2Generator,
                 out_file: Path,
                 **generate_kwargs):
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
        try:
            batch = generator.generate(prompt, max_len, **generate_kwargs)
        except RuntimeError as e:
            print("Error during generation with prompt:", prompt)
            print(e)
            print("Emptying CUDA cache and retrying...")
            torch.cuda.empty_cache()

            batch = ["GENERATION_ERROR_CUDA"] * len(prompt)

        for generation in batch:
            with out_file.open('a') as f:
                print(json.dumps(generation), file=f)
            yield generation


def gpt2_ctrl(prompts: pd.Series,
              max_len: int,
              num_samples: int,
              batch_size: int,
              prompt_ctrl_code: str,
              model_name_or_path: str,
              out_file: Path):
    # Use default gpt2 architecture
    generator = GPT2Generator(model_name_or_path)

    # Add some special tokens (inline metadata)
    with open(Path(model_name_or_path) / 'added_tokens.json') as f:
        ctrl_codes = list(json.load(f).keys())
    assert prompt_ctrl_code in ctrl_codes
    print('Added tokens:', ctrl_codes)
    num_tokens_added = generator.tokenizer.add_tokens(ctrl_codes)
    assert num_tokens_added == len(ctrl_codes)
    print("Tokenizer vocab size:", generator.tokenizer.vocab_size)

    # Prepend ctrl code to prompts
    prompts = prompt_ctrl_code + prompts
    print(prompts)

    yield from _gpt2_helper(prompts=prompts,
                            max_len=max_len,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            generator=generator,
                            out_file=out_file)


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

    yield from _gpt2_helper(prompts=prompts,
                            max_len=max_len,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            generator=generator,
                            out_file=out_file)


def gpt2(prompts: pd.Series,
         max_len: int,
         num_samples: int,
         batch_size: int,
         model_name_or_path: str,
         out_file: Path,
         **generate_kwargs) -> Iterable[str]:
    # Setup model
    generator = GPT2Generator(model_name_or_path)

    yield from _gpt2_helper(prompts=prompts,
                            max_len=max_len,
                            num_samples=num_samples,
                            batch_size=batch_size,
                            generator=generator,
                            out_file=out_file,
                            **generate_kwargs)


@click.command()
@click.argument('out_dir')
@click.option('--dataset_file', required=False, type=str)
@click.option('--eos_prompt/--no_eos_prompt', default=False)
@click.option('--model_type', required=True,
              type=click.Choice(['gpt2', 'ctrl', 'gpt2-affect', 'gpt2-ctrl',
                                 'pplm', 'gpt2-greedy', 'gpt2-naughty-list',
                                 'openai-gpt', 'xlnet']))
@click.option('--model_name_or_path', required=True)
@click.option('--perspective_rps', default=25)
@click.option('--gen_samples', default=25)
@click.option('--gen_max_len', default=20)
@click.option('--gen_batch_size', default=32)
@click.option('--shard', default=None, type=int)
@click.option('--num_shards', default=0)
@click.option('--resume/--no-resume', default=False)
def main(out_dir: str,
         eos_prompt: bool,
         dataset_file: Optional[str],
         model_type: str,
         model_name_or_path: str,
         perspective_rps: int,
         gen_samples: int,
         gen_max_len: int,
         gen_batch_size: int,
         shard: Optional[int],
         num_shards: Optional[int],
         resume: bool):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=resume)

    # Load dataset
    if eos_prompt:
        if model_type == 'gpt2' or model_type == 'gpt2-affect' or model_type == 'pplm' or model_type == 'gpt2-naughty-list':
            prompts = pd.Series('<|endoftext|>')
        elif model_type == 'gpt2-ctrl':
            prompts = pd.Series('<|nontoxic|>')
        elif model_type == 'ctrl':
            # HACK: update gen_samples since we use it as our batch size for pipelines
            prompts = pd.Series('').repeat(gen_samples // gen_batch_size + 1)
            gen_samples = gen_batch_size
        elif model_type == 'xlnet':
            prompts = pd.Series('<s>').repeat(gen_samples // gen_batch_size + 1)
            gen_samples = gen_batch_size
        elif model_type == 'openai-gpt':
            prompts = pd.Series('. ').repeat(gen_samples // gen_batch_size + 1)
            gen_samples = gen_batch_size
        else:
            raise RuntimeError('Model not implemented with EOS prompts')
    elif dataset_file:
        df = pd.read_csv(dataset_file)
        prompts = df['prompt.text']
    else:
        raise click.exceptions.MissingParameter('Missing dataset file or eos prompt option')

    # Select shard
    if num_shards:
        assert shard is not None and 0 <= shard < num_shards
        print("Using shard", shard, "of", num_shards)
        prompts = np.array_split(prompts, num_shards)[shard]
        generations_file = out_dir / f'generations_shard_{shard}_of_{num_shards - 1}.jsonl'
        perspective_file = out_dir / f'perspective_shard_{shard}_of_{num_shards - 1}.jsonl'
    else:
        print("Using entire dataset")
        generations_file = out_dir / 'generations.jsonl'
        perspective_file = out_dir / 'perspective.jsonl'

    print("Prompts:")
    print(prompts)

    # Create perspective worker thread
    perspective = PerspectiveWorker(out_file=perspective_file,
                                    total=len(prompts) * gen_samples,
                                    rps=perspective_rps)

    # Generate and request perspective scores
    if model_type == 'gpt2':
        generations_iter = gpt2(prompts=prompts,
                                max_len=gen_max_len,
                                num_samples=gen_samples,
                                batch_size=gen_batch_size,
                                model_name_or_path=model_name_or_path,
                                out_file=generations_file)
    elif model_type == 'gpt2-greedy':
        generations_iter = gpt2(prompts=prompts,
                                max_len=gen_max_len,
                                num_samples=gen_samples,
                                batch_size=gen_batch_size,
                                model_name_or_path=model_name_or_path,
                                out_file=generations_file,
                                sample=False)
    elif model_type == 'gpt2-naughty-list':
        # Load tokenized naughty words
        # FIXME: output dir must already exist with this file
        with open(out_dir / 'gpt2_naughty_token_ids.pkl', 'rb') as f:
            naughty_list_ids = pickle.load(f)
        generations_iter = gpt2(prompts=prompts,
                                max_len=gen_max_len,
                                num_samples=gen_samples,
                                batch_size=gen_batch_size,
                                model_name_or_path=model_name_or_path,
                                out_file=generations_file,
                                bad_words_ids=naughty_list_ids)
    elif model_type == 'gpt2-affect':
        generations_iter = gpt2_affect(prompts=prompts,
                                       max_len=gen_max_len,
                                       num_samples=gen_samples,
                                       batch_size=gen_batch_size,
                                       model_name_or_path=model_name_or_path,
                                       out_file=generations_file,
                                       # Affect
                                       target_class=0,
                                       num_classes=2,
                                       beta=1)
    elif model_type == 'gpt2-ctrl':
        generations_iter = gpt2_ctrl(prompts=prompts,
                                     max_len=gen_max_len,
                                     num_samples=gen_samples,
                                     batch_size=gen_batch_size,
                                     model_name_or_path=model_name_or_path,
                                     out_file=generations_file,
                                     # GPT2-CTRL
                                     prompt_ctrl_code='<|nontoxic|>')
    elif model_type == 'openai-gpt':
        generations_iter = openai_gpt(prompts=prompts,
                                      max_len=gen_max_len,
                                      num_samples=gen_samples,
                                      model_name_or_path=model_name_or_path,
                                      out_file=generations_file)
    elif model_type == 'ctrl':
        assert model_name_or_path == 'ctrl'
        generations_iter = ctrl(prompts=prompts,
                                max_len=gen_max_len,
                                num_samples=gen_samples,
                                model_name_or_path=model_name_or_path,
                                out_file=generations_file,
                                # CTRL
                                ctrl_code='Wikipedia',
                                temperature=1.0,
                                repetition_penalty=1.2)
    elif model_type == 'xlnet':
        assert model_name_or_path == 'xlnet-base-cased'
        generations_iter = xlm(prompts=prompts,
                               max_len=gen_max_len,
                               batch_size=gen_batch_size,
                               num_samples=gen_samples,
                               out_file=generations_file)
    elif model_type == 'pplm':
        generations_iter = pplm(prompts=prompts,
                                max_len=gen_max_len,
                                num_samples=gen_samples,
                                batch_size=gen_batch_size,
                                class_label=0,
                                num_iterations=10,
                                model_name_or_path='toxicity',
                                out_file=generations_file)
    else:
        raise NotImplementedError(f'Model {model_name_or_path} not implemented')

    for i, gen in enumerate(generations_iter):
        perspective(f'generation-{i}', gen)

    torch.cuda.empty_cache()

    perspective.stop()
    print('Done')


if __name__ == '__main__':
    main()
