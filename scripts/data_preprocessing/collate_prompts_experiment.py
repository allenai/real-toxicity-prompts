from typing import Any, Dict, Iterable, Optional, List

import click
import pandas as pd
from tqdm import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.perspective_api import unpack_scores
from utils.utils import batchify, load_jsonl


def format_response(text: str, response: dict):
    if response['response']:
        response = unpack_scores(response['response'])[0]
    else:
        response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
    return {'text': text, **response}


def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        yield format_response(generation, response)


def collate(dataset: Optional[pd.DataFrame], generations: List[str], responses: Iterable[Dict[str: Any]],
            output_file: str):
    assert len(generations) % len(dataset) == 0
    n = len(generations) // len(dataset)
    print(f"Detected samples per prompt:", n)

    generations_col = list(tqdm(batchify(make_generations_col(generations, responses), n)))

    if dataset:
        dataset['generations'] = generations_col
    else:
        # TODO: check that this works
        dataset = pd.Series(generations_col)

    dataset.to_json(output_file, orient='records', lines=True)


@click.command()
@click.option('--eos', required=False)
@click.option('--prompts_file', required=False)
@click.option('--generations_file', required=True)
@click.option('--perspective_file', required=True)
@click.argument('--output_file')
def collate_prompts_experiment(eos: bool, dataset_file: str, generations_file: str, perspective_file: str,
                               output_file: str):
    assert eos or dataset_file
    prompts = pd.read_json(dataset_file, lines=True)
    generations = pd.read_json(generations_file, lines=True)
    responses = load_jsonl(perspective_file)
    collate(prompts, generations, responses, output_file)


if __name__ == '__main__':
    collate_prompts_experiment()
