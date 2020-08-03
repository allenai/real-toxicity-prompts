from math import ceil
from utils.constants import OPENAI_API_KEY
import openai
from pathlib import Path
import pandas as pd
import json
import click
from tqdm.auto import tqdm, trange

from utils.utils import batchify

openai.api_key = OPENAI_API_KEY

PROMPT_BATCH_SIZE = 1
ENGINE = 'davinci'
MAX_TOKENS = 20
N = 20


def generate_eos(num_samples: int):
    for _ in trange(ceil(num_samples / N)):
        while True:
            try:
                yield openai.Completion.create(
                    engine=ENGINE,
                    prompt='',
                    max_tokens=MAX_TOKENS,
                    n=N
                )
                break
            except openai.error.APIError as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")


def generate(prompts):
    for prompt in tqdm(batchify(prompts, PROMPT_BATCH_SIZE), total=ceil(len(prompts) / PROMPT_BATCH_SIZE)):
        while True:
            try:
                yield openai.Completion.create(
                    engine=ENGINE,
                    prompt=prompt,
                    max_tokens=MAX_TOKENS,
                    n=N
                )
                break
            except Exception as e:
                tqdm.write(str(e))
                tqdm.write("Retrying...")


def write(response, fp):
    json.dump(response.to_dict_recursive(), fp)
    fp.write('\n')


@click.command()
@click.option('--out_file', required=True)
@click.option('--prompts_file', type=str, default=None)
@click.option('--num_eos_samples', type=int, default=None)
def main(prompts_file: str, num_eos_samples: int, out_file: str):
    assert prompts_file or num_eos_samples

    out_file = Path(out_file)

    num_skip = 0
    if out_file.exists():
        # FIXME: relies on batch size 1
        generations = pd.read_json(out_file, lines=True)
        num_skip = len(generations)
        print(num_skip, "previously completed generations")

    with open(out_file, 'a') as f:
        if prompts_file:
            print("Loading prompts from", prompts_file)
            df = pd.read_csv(prompts_file)
            prompts = df['prompt.text']
            for gen in generate(prompts[num_skip:]):
                write(gen, f)
        else:
            print("Using EOS as prompt")
            for gen in generate_eos(num_eos_samples - num_skip):
                write(gen, f)


if __name__ == '__main__':
    main()
