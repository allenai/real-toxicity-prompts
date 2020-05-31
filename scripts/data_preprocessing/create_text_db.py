import re
from pathlib import Path

import click
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

from utils.webtext_db import Doc, DocScore, SpanScore
from utils.utils import make_corpus_iter, batchify

Session = sessionmaker()

WT_SIZE = 8_282_020
NUM_SHARDS = 20
SHARD_SIZE = WT_SIZE // NUM_SHARDS


@click.command()
@click.option('--corpus_dir', required=True)
@click.option('--batch_size', default=1000)
@click.argument('database_file')
def main(corpus_dir: str, database_file: str, batch_size: int):
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise click.ClickException('Corpus dir does not exist')

    database_file = Path(database_file)
    if database_file.exists():
        raise click.ClickException('Database already exists')

    engine = create_engine(f'sqlite:///{database_file}', echo=False)

    # Create schemas
    print('Creating schemas...')
    Doc.metadata.create_all(engine)
    DocScore.metadata.create_all(engine)
    SpanScore.metadata.create_all(engine)
    print("Note: DocScores and SpanScores must be loaded manually")

    # Start session
    Session.configure(bind=engine)
    session = Session()

    print('Reading texts into database...')
    wt_iter = make_corpus_iter(corpus_dir)
    pbar = tqdm(total=WT_SIZE)
    try:
        for batch in batchify(enumerate(wt_iter), batch_size=batch_size):
            rows = []
            for i, (doc_id, text) in batch:
                # Verify that index is correct
                shard, idx = doc_id.split('-')
                shard = int(re.search(r'[0-9]+', shard)[0])
                idx = int(idx)
                assert i == shard * SHARD_SIZE + idx

                # Create location string
                location = f'{shard}-{idx}'
                doc = Doc(id=i, location=location, text=text)
                rows.append(doc)

            # Update pbar
            pbar.update(len(batch))

            # Save objects and clear session
            session.bulk_save_objects(rows)
            session.commit()
            session.expunge_all()
    except Exception as e:
        print(e)
        session.rollback()
    finally:
        session.close()


if __name__ == '__main__':
    main()
