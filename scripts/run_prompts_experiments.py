import logging
import logging
import pickle
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from generation.generation import gpt2, gpt2_affect, gpt2_ctrl, openai_gpt, ctrl, xlm, pplm
from utils.perspective_api import PerspectiveAPI
from utils.utils import load_cache

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
        api = PerspectiveAPI(rate_limit=rps)
        pbar = tqdm(total=total, dynamic_ncols=True, position=1)
        api.request_bulk(queue_iter, output_file=responses_file, pbar=pbar)


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
