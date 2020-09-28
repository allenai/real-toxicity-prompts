import pandas as pd
import sqlite3

from utils.constants import DATA_DIR

from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
import click
import spacy


@click.command()
@click.argument('output-dir')
def main(output_dir, model="en_core_web_sm", n_jobs=8, batch_size=100_000):
    output_dir = Path(output_dir)

    nlp = spacy.load(model, disable=['parser', 'tagger', 'ner'])  # load spaCy model
    print("Loaded model '%s'" % model)

    if not output_dir.exists():
        output_dir.mkdir()

    # load and pre-process the OWTC corpus
    print("Loading OWTC data...")
    owtc = DATA_DIR / 'openwebtext.db'

    con = sqlite3.connect(owtc, check_same_thread=False)
    partitions = pd.read_sql('SELECT SUBSTR(docs.location, 9) AS md5_hash, docs.text FROM docs',
                             con=con,
                             chunksize=batch_size)

    print("Processing texts...")
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
    do = delayed(partial(transform_texts, nlp))
    tasks = (do(i, batch, output_dir) for i, batch in enumerate(partitions))
    executor(tasks)


def transform_texts(nlp, batch_id, df, output_dir):
    print(nlp.pipe_names)

    out_path = Path(output_dir) / ("%d.csv" % batch_id)
    if out_path.exists():  # return None in case same batch is called again
        return None

    print("Processing batch", batch_id)

    df['word_count'] = list(map(len, nlp.pipe(df['text'])))
    df[['md5_hash', 'word_count']].to_csv(out_path)


if __name__ == '__main__':
    main()
