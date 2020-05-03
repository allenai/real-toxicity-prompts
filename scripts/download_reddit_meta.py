import json
from pathlib import Path

import pandas as pd
from psaw import PushshiftAPI
from tqdm import tqdm

from utils.constants import DATA_DIR

urls_file = DATA_DIR / 'openwebtext-urls.csv'
urls = pd.read_csv(urls_file)

subsample_dir = DATA_DIR / 'subsample-100k'
docs = pd.read_csv(subsample_dir / 'docs.csv')
docs = docs.merge(urls)

api = PushshiftAPI()

out_file = Path('reddit.jsonl')
if out_file.exists():
    with out_file.open() as f:
        num_lines = sum(1 for line in f)

with out_file.open('a') as f:
    for url in tqdm(docs.url[num_lines:]):
        response = [x.d_ for x in api.search_submissions(url=url)]
        print(json.dumps(response), file=f)
