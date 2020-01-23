import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from pathlib import Path

from constants import DATA_DIR, TEXTS_DIR, OUTPUT_DIR


def save_finetune_data(df: pd.DataFrame, train_file: Path, val_file: Path):
    text_data = df['id'].apply(lambda fn: (TEXTS_DIR / (fn + ".txt")).read_text())
    train, val = train_test_split(text_data, test_size=0.2, random_state=42)

    print('\n'.join(train), file=train_file.open('w'))
    print('\n'.join(val), file=val_file.open('w'))


if __name__ == '__main__':
    # Create sql connection
    database_path = DATA_DIR / 'small_perspective_api_responses.db'
    engine = create_engine(f'sqlite:///{database_path}', echo=False)

    # Load dataframe from sql
    df = pd.read_sql(
        'select * from responses where toxicity < 0.1',
        con=engine
    )



    # save_finetune_data(df, )
