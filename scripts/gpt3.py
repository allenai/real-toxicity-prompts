import math
from utils.constants import OPENAI_API_KEY
import openai
from pathlib import Path
import pandas as pd
import json
import click
from tqdm.auto import tqdm

from utils.utils import batchify

openai.api_key = OPENAI_API_KEY

BATCH_SIZE = 2
ENGINE = 'davinci'
MAX_TOKENS = 20
N = 25


def generate_eos(num_samples: int):
    for i in range(math.ceil(num_samples / BATCH_SIZE)):
        yield openai.Completion.create(
            engine=ENGINE,
            prompt='',
            max_tokens=MAX_TOKENS,
            n=25 * BATCH_SIZE
        )


def generate(prompts):
    for prompt in batchify(prompts, BATCH_SIZE):
        yield openai.Completion.create(
            engine=ENGINE,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            n=N
        )


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
    assert not out_file.exists()
    with open(out_file, 'w') as f:
        if prompts_file:
            print("Loading prompts from", prompts_file)
            df = pd.read_csv(prompts_file)
            prompts = df['prompt.text']
            for gen in tqdm(generate(prompts), total=len(prompts) / BATCH_SIZE):
                write(gen, f)
        else:
            print("Using EOS as prompt")
            for gen in tqdm(generate_eos(num_eos_samples), total=num_eos_samples / BATCH_SIZE):
                write(gen, f)


if __name__ == '__main__':
    main()
