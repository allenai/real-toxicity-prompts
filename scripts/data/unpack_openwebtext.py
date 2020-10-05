import tempfile
from pathlib import Path
from shutil import unpack_archive, rmtree
import multiprocessing as mp
from itertools import chain

import numpy as np
import click
from joblib import Parallel, delayed, dump
from tqdm.auto import tqdm


def text_files_in_dir(s):
    return [f for f in s.iterdir() if f.suffix == '.txt']


def read_text(file):
    return file.read_text()


def shardify_openwebtext(input_dir: str, output_dir: str, n_processes: int, n_shards: int = 20):
    input_dir = Path(input_dir)
    assert input_dir.exists()

    output_dir = Path(output_dir)
    output_dir.mkdir()

    subset_dirs = [subset_dir for subset_dir in input_dir.iterdir() if subset_dir.is_dir()]
    with mp.Pool(processes=n_processes) as pool:
        # Get list of sorted files in OpenWebText
        text_files = sorted(list(chain.from_iterable(pool.map(text_files_in_dir, subset_dirs))))

        # Split list of files into shards
        shard_files = np.array_split(text_files, n_shards)

        # Save shards and associated filenames
        for i, split in enumerate(tqdm(shard_files)):
            tqdm.write("Loading text in shard...")
            shard = pool.map(read_text, split)

            tqdm.write("Saving shard...")
            shard_name = f'owtc{i:02d}'
            dump(shard, output_dir / f'{shard_name}.joblib')
            filenames = map(lambda f: f.stem, split)
            with open(output_dir / f'{shard_name}_filenames.txt', 'w') as f:
                print(*filenames, file=f, sep='\n')


@click.command()
@click.option('--archive', required=True)
@click.option('--n_jobs', default=16)
@click.argument('out_dir')
def unpack_openwebtext(archive: str, out_dir: str, n_jobs: int):
    out_dir = Path(out_dir)
    out_dir.mkdir()

    tmp_dir = Path(tempfile.mkdtemp(prefix='openwebtext'))
    print("Unpacking subset archives to", tmp_dir)
    unpack_archive(archive, extract_dir=tmp_dir)
    subset_tarfiles = [x for x in (tmp_dir / 'openwebtext').iterdir()]

    print("Unpacking corpus to", out_dir)
    subset_out_dirs = [out_dir / subset.stem for subset in subset_tarfiles]
    Parallel(n_jobs=n_jobs)(
        delayed(unpack_archive)(tarfile_path, subset_out_dir, 'xztar')
        for tarfile_path, subset_out_dir in zip(subset_tarfiles, subset_out_dirs)
    )

    rmtree(tmp_dir)


if __name__ == '__main__':
    unpack_openwebtext()
