import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from pathlib import Path
from util import load_text
from run_lm_finetuning import main
import sys
from os import makedirs

from constants import DATA_DIR, TEXTS_DIR

RANDOM_STATE = 42


def save_data(experiment_dir: Path, dataset: pd.DataFrame, name: str, eos_token='<|endoftext|>'):
    metadata_file = experiment_dir / f'{name}.pkl'
    dataset.to_pickle(metadata_file)

    text = dataset['id'].apply(lambda id: load_text(id + ".txt"))
    text_file = experiment_dir / f'{name}.txt'
    print(*text, sep=eos_token, file=text_file.open('w'))
    
    return text_file


def train_model(output_dir: Path, train_file: Path, val_file: Path, batch_size=1, gradient_accumulation_steps=4):
    sys.argv = [
        "run_lm_finetuning.py",
        f'--output_dir={output_dir}',
        '--model_type=gpt2',
        '--model_name_or_path=gpt2',
        '--do_train',
        f'--train_data_file={train_file}',
        '--do_eval',
        f'--eval_data_file={val_file}',
        # Some hacks to get it to work on my poor 1080Ti
        f'--per_gpu_train_batch_size={batch_size}',
        f'--per_gpu_eval_batch_size={batch_size}',
        f'--gradient_accumulation_steps={gradient_accumulation_steps}'
    ]
    main()
    
    
def run_experiment(query: str, engine: Engine, experiments_dir: Path, experiment_name: str, test_size=0.1):
    print("Running experiment with query:", query)
    
    # Load dataframe from sql
    print("Querying database...")
    df = pd.read_sql(query, con=engine)
    print("Number of examples returned from query:", len(df))

    # Save training and evaluation data
    print("Saving training data...")
    experiment_dir = experiments_dir / experiment_name
    makedirs(experiment_dir)
    train, val = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
    train_file = save_data(experiment_dir, train, 'train')
    val_file = save_data(experiment_dir, val, 'val')
    
    print("Training model...")
    experiment_output_dir = experiment_dir / 'finetune_output'
    train_model(experiment_output_dir, train_file, val_file)


if __name__ == '__main__':
    # Create sql connection
    database_path = DATA_DIR / 'perspective_api_responses.db'
    engine = create_engine(f'sqlite:///{database_path}', echo=False)
    
    experiments = (
        ('toxicity < 0.01', 'finetune_toxicity_lt1'),
#         ('toxicity < 0.10', 'finetune_toxicity_lt10'),
#         ('toxicity > 0.99', 'finetune_toxicity_gt99'),
#         ('toxicity > 0.90', 'finetune_toxicity_gt90')
    )
    
    experiments_dir = Path() / 'experiments'

    for predicate, experiment_name in experiments:
        # TODO: REMOVE LIMIT
        sample_query = f'select * from responses where {predicate} limit 100'
        run_experiment(sample_query, engine, experiments_dir, experiment_name)
