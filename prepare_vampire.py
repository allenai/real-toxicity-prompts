import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import pandas as pd

from constants import DATA_DIR, URLS_PICKLE, BIAS_PICKLE, TEXTS_DIR, VAMPIRE_DIR

allsides = pd.read_pickle(BIAS_PICKLE)
urls = pd.read_pickle(URLS_PICKLE)
urls_merged = urls.merge(allsides)

for i, row in tqdm(urls_merged.iterrows(), total=len(urls_merged)):
    text_file = TEXTS_DIR / row.filename
    out_dict = {
        'text': text_file.read_text(),
        'label': row.bias
    }
    
    out_file = VAMPIRE_DIR / text_file.with_suffix('.vampire.json').name
    with out_file.open('w') as f:
        json.dump(out_dict, f)
