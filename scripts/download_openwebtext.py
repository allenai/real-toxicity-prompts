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

    subset_dir = tmp_dir / 'openwebtext'
    subset_tarfiles = [x for x in subset_dir.iterdir()]

    print("Unpacking corpus to", out_dir)
    Parallel(n_jobs=16)(
        delayed(unpack_archive)(tarfile_path, out_dir, 'xztar')
        for tarfile_path in subset_tarfiles
    )

    rmtree(tmp_dir)


if __name__ == '__main__':
    unpack_openwebtext()
