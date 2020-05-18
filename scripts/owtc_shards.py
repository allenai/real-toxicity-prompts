from typing import List

from tqdm.auto import tqdm
from pathlib import Path
import numpy as np

from utils.constants import DATA_DIR

EOS = 50265


def load_filenames(shard_files: List[Path]):
    return [f.with_suffix('.files.tmp').read_text().split() for f in shard_files]


def create_npy_shards(tmp_dirs, out_dir):
    num_docs = 0
    shards = {}
    for t in tmp_dirs:
        bpe_files = [x for x in t.iterdir() if x.suffixes == ['.tmp']]
        for b in bpe_files:
            shards.setdefault(b.stem, []).append(b)

    for name, shard_files in tqdm(shards.items(), desc='Shard'):
        shard_arr = []
        filenames = []
        for shard_file, shard_filenames in zip(shard_files, load_filenames(shard_files)):
            with shard_file.open() as f:
                for line in tqdm(f, desc=str(shard_file), total=len(shard_filenames)):
                    toks = line.split()
                    toks.append(EOS)
                    toks_arr = np.array(toks, dtype=int)
                    assert np.count_nonzero(toks_arr == EOS) == 1  # only one EOS per document
                    shard_arr.append(toks_arr)
                    num_docs += 1

            filenames.extend(shard_filenames)

        assert len(shard_arr) == len(filenames)

        # Save documents
        shard_arr = np.concatenate(shard_arr)
        np.savez(out_dir / f'owtc{name}_tokens', shard_arr)

        # Save filenames
        with open(out_dir / f'owtc{name}_filenames.txt', 'w') as f:
            print(*filenames, file=f, sep='\n')

    return num_docs


def main():
    bpe_dir = DATA_DIR / 'openwebtext_bpe'

    tmp_dirs = [bpe_dir / 'owtc_shards_v2_1', bpe_dir / 'owtc_shards_v2_2']
    out_dir = bpe_dir

    expected_total = 8003023
    actual_total = create_npy_shards(tmp_dirs, out_dir)
    assert expected_total == actual_total


if __name__ == '__main__':
    main()
