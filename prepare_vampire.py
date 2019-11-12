import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import pandas as pd

from constants import DATA_DIR, URLS_PICKLE, BIAS_PICKLE, TEXTS_DIR, VAMPIRE_DIR

NUM_THREADS = 8

allsides = pd.read_pickle()
urls = pd.read_pickle(URLS_PICKLE)
urls_merged = urls.merge(BIAS_PICKLE)


def vampire_file(file):
    out_dict = {'text': file.read_text()}
    filename = file.name
    
    if filename in urls_merged.index:
        out_dict['label'] = urls_merged.loc[filename, 'bias']
    
    out_file = VAMPIRE_DIR / file.with_suffix('.json').name
    with out_file.open('w') as f:
        json.dump(out_dict, f)


with tqdm(total=len(urls_merged)) as t:
    pool = ThreadPool(NUM_THREADS)
    for _ in pool.imap_unordered(vampire_file, TEXTS_DIR.iterdir()):
        t.update(1)
        