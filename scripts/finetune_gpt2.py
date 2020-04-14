import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy.engine import Engine
from pathlib import Path
from utils.utils import load_text
from scripts.transformers import run_language_modeling
import sys
from os import makedirs

from utils.db import perspective_db_engine

RANDOM_STATE = 42
COLUMN_NAME = 'filename'


def save_data(experiment_dir: Path, dataset: pd.DataFrame, name: str, eos_token='<|endoftext|>'):
    metadata_file = experiment_dir / f'{name}.pkl'
    dataset.to_pickle(metadata_file)

    text = dataset[COLUMN_NAME].apply(lambda filename: load_text(filename))
    text_file = experiment_dir / f'{name}.txt'
    print(*text, sep=eos_token, file=text_file.open('w'))

    return text_file


def train_model(output_dir: Path,
                train_file: Path,
                val_file: Path,
                epochs: int,
                batch_size=1,
                gradient_accumulation_steps=4,
                eval_steps=5049):
    # Update eval steps for gradient accumulation
    eval_steps //= gradient_accumulation_steps

    sys.argv = [
        "run_language_modeling.py",
        f'--output_dir={output_dir}',
        '--model_type=gpt2',
        '--model_name_or_path=gpt2',
        '--do_train',
        f'--train_data_file={train_file}',
        f'--num_train_epochs={epochs}',
        '--do_eval',
        f'--eval_data_file={val_file}',
        # More reasonable logging and checkpointing
        '--evaluate_during_training',
        f'--log_after_epoch',
        f'--save_after_epoch',
        f'--patience={2}',
        # Some hacks to get it to work on my poor 1080Ti
        f'--per_gpu_train_batch_size={batch_size}',
        f'--per_gpu_eval_batch_size={batch_size}',
        f'--gradient_accumulation_steps={gradient_accumulation_steps}',
    ]
    run_language_modeling.main()


def run_experiment(query: str, engine: Engine, experiments_dir: Path, experiment_name: str, test_size=0.1, limit=None,
                   epochs=1):
    print("Running experiment with query:", query)

    # Load dataframe from sql
    print("Querying database...")
    df = pd.read_sql(query, con=engine)
    print("Number of examples returned from query:", len(df))
    print()

    # Save training and evaluation data
    print("Saving training data...")
    experiment_dir = experiments_dir / experiment_name
    makedirs(experiment_dir)
    (experiment_dir / 'query.txt').write_text(query + '\n')

    # Split dataset
    if limit:
        num_examples = min(limit, len(df))
        train_size = int((1 - test_size) * num_examples)
        test_size = int(test_size * num_examples)
        print("Rows limited to", limit)
    else:
        train_size = None
    print("Train size: ", train_size, ", Test size:", test_size)
    train, val = train_test_split(df, train_size=train_size, test_size=test_size, random_state=RANDOM_STATE)

    train_file = save_data(experiment_dir, train, 'train')
    val_file = save_data(experiment_dir, val, 'val')
    print()

    print("Training model...")
    experiment_output_dir = experiment_dir / 'finetune_output'
    train_model(experiment_output_dir, train_file, val_file, epochs=epochs)


def main():
    # Create sql connection
    engine = perspective_db_engine()
    conn = engine.connect()

    responses_percentiles_statement = """
    CREATE TEMP TABLE responses_percentiles AS
    SELECT responses.filename  AS filename,
           NTILE(100) OVER win AS p
    FROM responses
        WINDOW
            win AS (ORDER BY responses.toxicity);
    """
    conn.execute(responses_percentiles_statement)

    experiments_dir = Path() / 'experiments'
    experiments = (
        ("SELECT * FROM responses_percentiles WHERE responses_percentiles.p <= 2", 'finetune_toxicity_percentile_lte2',
         None),
        ("SELECT * FROM responses_percentiles WHERE responses_percentiles.p >= 40 AND responses_percentiles.p < 60",
         'finetune_toxicity_percentile_middle_20_subsample', 155408),
        (
            "SELECT * FROM responses_percentiles WHERE responses_percentiles.p >= 99",
            'finetune_toxicity_percentile_gte99',
            None),
    )

    for query, experiment_name, limit in experiments:
        run_experiment(query, conn, experiments_dir, experiment_name, limit=limit, epochs=100)


if __name__ == '__main__':
    main()
