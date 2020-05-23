import tempfile
from pathlib import Path
from shutil import unpack_archive, rmtree

import click
from joblib import Parallel, delayed


@click.command()
@click.option('--archive', required=True)
@click.argument('out_dir')
def unpack_openwebtext(archive: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir()

    tmp_dir = Path(tempfile.mkdtemp(prefix='openwebtext'))
    print("Unpacking subset archives to", tmp_dir)
    unpack_archive(archive, extract_dir=tmp_dir)
    subset_tarfiles = [x for x in (tmp_dir / 'openwebtext').iterdir()]

    print("Unpacking corpus to", out_dir)
    subset_out_dirs = [out_dir / subset.stem for subset in subset_tarfiles]
    Parallel(n_jobs=16)(
        delayed(unpack_archive)(tarfile_path, subset_out_dir, 'xztar')
        for tarfile_path, subset_out_dir in zip(subset_tarfiles, subset_out_dirs)
    )

    rmtree(tmp_dir)


if __name__ == '__main__':
    unpack_openwebtext()
