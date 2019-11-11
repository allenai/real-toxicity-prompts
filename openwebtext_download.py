import os
from pathlib import Path
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import tarfile

from constants import DATA_DIR, OPENWEBTEXT_ARCHIVES_DIR, TEXTS_DIR


NUM_THREADS = 16


# Extract text files from OpenWebText
def extract_tar(tarfile_path):
    # TODO: check whether an archive has already been extracted
    archive = tarfile.open(tarfile_path)
    archive.extractall(TEXTS_DIR)
    archive.close()


def find_pending_archives():
    subset_archive_paths = set(OPENWEBTEXT_ARCHIVES_DIR.iterdir())
    existing_text_files = set([file.name for file in TEXTS_DIR.iterdir()])
    
    pending_archive_paths = []
    for archive_path in subset_archive_paths:
        first_text_file = tarfile.open(archive_path).getnames()[0]
        if first_text_file not in existing_text_files:
            pending_archive_paths.append(archive_path)
    
    return pending_archive_paths
    

def main():
    # Download and un-tar OpenWebText archive
    openwebtext_archive_file = DATA_DIR / 'openwebtext.tar.xz'

    if not OPENWEBTEXT_ARCHIVES_DIR.exists():
        if not openwebtext_archive_file.exists():
            # Download the dataset
            !pip install gdown
            !gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
            assert openwebtext_archive_file.exists()

        # Un-archive the dataset
        !tar -xf {openwebtext_archive_file}
        assert OPENWEBTEXT_ARCHIVES_DIR.exists()

    pending_archive_paths = find_pending_archives()
    with tqdm(total=len(pending_archive_paths)) as t:
        pool = ThreadPool(NUM_THREADS)
        for _ in pool.imap_unordered(extract_tar, pending_archive_paths):
            t.update(1)
