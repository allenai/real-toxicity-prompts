import json
import math
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from datasets.openwebtext_dataset import openwebtext_dataloader
from utils.constants import OUTPUT_DIR


def write_examples(df: pd.DataFrame, out_file: Path):
    texts = openwebtext_dataloader(df.filename)
    with out_file.open('a') as f:
        for (filename, text), control_code in tqdm(zip(texts, df.control_code), total=len(df)):
            print(json.dumps({'control_code': control_code, 'text': text}), file=f)


def make_dataframe(experiment_dirs: List[Path], control_codes: List[str], dataset: str):
    min_len = math.inf
    dataframes = []
    for e, c in zip(experiment_dirs, control_codes):
        df = pd.read_pickle(e / f'{dataset}.pkl')
        df['control_code'] = c
        dataframes.append(df)
        min_len = min(min_len, len(df))

    samples_per_experiment = min_len // len(experiment_dirs)
    sampled_df = pd.concat([df.sample(n=samples_per_experiment) for df in dataframes])
    return sampled_df


def main():
    experiment_dirs = [
        '/data/language-model-toxicity/models/gpt2-finetuned-models/finetune_toxicity_percentile_gte99',
        '/data/language-model-toxicity/models/gpt2-finetuned-models/finetune_toxicity_percentile_middle_20_subsample',
        '/data/language-model-toxicity/models/gpt2-finetuned-models/finetune_toxicity_percentile_lte2'
    ]
    experiment_dirs = [Path(e) for e in experiment_dirs]

    control_codes = [
        'toxic',
        'neutral',
        'safe'
    ]

    out_dir = OUTPUT_DIR / 'ctrl_data'
    assert all(e.exists() for e in experiment_dirs) and not out_dir.exists()
    out_dir.mkdir()

    train_df = make_dataframe(experiment_dirs, control_codes, 'train')
    train_df.to_pickle(out_dir / 'train.pkl')
    val_df = make_dataframe(experiment_dirs, control_codes, 'val')
    val_df.to_pickle(out_dir / 'val.pkl')

    write_examples(train_df, out_dir / 'train.jsonl')
    write_examples(val_df, out_dir / 'val.jsonl')


if __name__ == '__main__':
    main()
