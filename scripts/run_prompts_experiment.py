import pickle
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import torch

from generation.generation import gpt2, gpt2_affect, gpt2_ctrl, openai_gpt, ctrl, xlm, pplm
from scripts.data_preprocessing.collate_prompts_experiment import collate
from utils.perspective_api import PerspectiveWorker
from utils.utils import load_jsonl

ALLOWED_MODELS = ['gpt2', 'gpt2-affect', 'gpt2-ctrl', 'gpt2-greedy', 'gpt2-naughty-list',
                  'pplm', 'ctrl', 'openai-gpt', 'xlnet']


@click.command()
@click.argument('output-dir')
@click.option('--dataset-file', required=False, type=str,
              help='JSONL file containing prompts data. Each row must contain a prompt at `row["prompt"]["text"]`.')
@click.option('--use-eos/--use-dataset', default=False, help='Whether to use EOS or a dataset file for generation.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--model-type', required=True,
              type=click.Choice(ALLOWED_MODELS))
@click.option('--perspective-rate-limit', default=25)
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=20, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=32)
@click.option('--resume/--no-resume', default=False)
def main(output_dir: str, dataset_file: Optional[str], use_eos: bool, model: str, model_type: str, n: int,
         max_tokens: int, batch_size: int, perspective_rate_limit: int, resume: bool):
    # Load prompts
    if dataset_file:
        assert not use_eos
        # Load prompts from dataset file
        assert dataset_file.endswith('.jsonl')
        dataset = pd.read_json(dataset_file, lines=True)
        prompts = pd.json_normalize(dataset['prompt'])['text']
    elif use_eos:
        assert not dataset_file
        dataset = None
        # Create EOS prompts
        if model_type in ['gpt2', 'gpt2-affect', 'gpt2-naughty-list', 'pplm']:
            prompts = pd.Series('<|endoftext|>')
        elif model_type == 'gpt2-ctrl':
            prompts = pd.Series('<|nontoxic|>')
        elif model_type == 'ctrl':
            # HACK: update gen_samples since we use it as our batch size for pipelines
            prompts = pd.Series('').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'xlnet':
            prompts = pd.Series('<s>').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'openai-gpt':
            prompts = pd.Series('. ').repeat(n // batch_size + 1)
            n = batch_size
        elif model_type == 'gpt3':
            prompts = pd.Series('').repeat(n // batch_size + 1)
        else:
            raise RuntimeError('Model not implemented with EOS prompts')
    else:
        raise click.exceptions.MissingParameter('Missing --dataset-file or --use-eos option.')
    print('Prompts:', '\n', prompts)

    # Create output files
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=resume)
    generations_file = output_dir / 'generations.jsonl'
    perspective_file = output_dir / 'perspective.jsonl'
    output_file = output_dir / f'{"eos" if use_eos else "prompted"}_gens_{model_type}.jsonl'

    # Create perspective worker thread
    perspective = PerspectiveWorker(out_file=perspective_file,
                                    total=len(prompts) * n,
                                    rate_limit=perspective_rate_limit)

    # Setup model for generation
    # TODO: move this logic into generation.py
    if model_type == 'gpt2':
        generations_iter = gpt2(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                batch_size=batch_size,
                                model_name_or_path=model,
                                out_file=generations_file)
    elif model_type == 'gpt2-greedy':
        print("Using n=1 for greedy generation (sampling does not apply)")
        generations_iter = gpt2(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=1,
                                batch_size=batch_size,
                                model_name_or_path=model,
                                out_file=generations_file,
                                sample=False)
    elif model_type == 'gpt2-naughty-list':
        # Load pre-tokenized naughty words
        # FIXME: output dir must already exist with this file
        with open(output_dir / 'gpt2_naughty_token_ids.pkl', 'rb') as f:
            naughty_list_ids = pickle.load(f)
        generations_iter = gpt2(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                batch_size=batch_size,
                                model_name_or_path=model,
                                out_file=generations_file,
                                bad_words_ids=naughty_list_ids)
    elif model_type == 'gpt2-affect':
        generations_iter = gpt2_affect(prompts=prompts,
                                       max_len=max_tokens,
                                       num_samples=n,
                                       batch_size=batch_size,
                                       model_name_or_path=model,
                                       out_file=generations_file,
                                       # Affect
                                       target_class=0,
                                       num_classes=2,
                                       beta=1)
    elif model_type == 'gpt2-ctrl':
        generations_iter = gpt2_ctrl(prompts=prompts,
                                     max_len=max_tokens,
                                     num_samples=n,
                                     batch_size=batch_size,
                                     model_name_or_path=model,
                                     out_file=generations_file,
                                     # GPT2-CTRL
                                     prompt_ctrl_code='<|nontoxic|>')
    elif model_type == 'openai-gpt':
        generations_iter = openai_gpt(prompts=prompts,
                                      max_len=max_tokens,
                                      num_samples=n,
                                      model_name_or_path=model,
                                      out_file=generations_file)
    elif model_type == 'ctrl':
        assert model == 'ctrl'
        generations_iter = ctrl(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                model_name_or_path=model,
                                out_file=generations_file,
                                # CTRL
                                ctrl_code='Wikipedia',
                                temperature=1.0,
                                repetition_penalty=1.2)
    elif model_type == 'xlnet':
        assert model == 'xlnet-base-cased'
        generations_iter = xlm(prompts=prompts,
                               max_len=max_tokens,
                               batch_size=batch_size,
                               num_samples=n,
                               out_file=generations_file)
    elif model_type == 'pplm':
        generations_iter = pplm(prompts=prompts,
                                max_len=max_tokens,
                                num_samples=n,
                                batch_size=batch_size,
                                class_label=0,
                                num_iterations=10,
                                model_name_or_path='toxicity',
                                out_file=generations_file)
    else:
        raise NotImplementedError(f'Model {model} not implemented')

    # Generate and collate perspective scores
    generations = []
    for i, gen in enumerate(generations_iter):
        generations.append(gen)
        perspective(f'generation-{i}', gen)

    torch.cuda.empty_cache()
    perspective.stop()
    print('Finished generation and perspective scoring!')

    print('Collating output files')
    collate(dataset, generations, load_jsonl(perspective_file), output_file)


if __name__ == '__main__':
    main()
