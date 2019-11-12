import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import pandas as pd

from constants import DATA_DIR, URLS_PICKLE, BIAS_PICKLE, TEXTS_DIR, VAMPIRE_DIR


def write_data_file(df, f):
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text_file = TEXTS_DIR / row.filename
        out_dict = {
            'text': text_file.read_text(),
            'label': row.bias
        }

        json.dump(out_dict, f)
        f.write('\n')


allsides = pd.read_pickle(BIAS_PICKLE)
urls = pd.read_pickle(URLS_PICKLE)
urls_merged = urls.merge(allsides)

data_shuffled = urls_merged.sample(frac=1)
split_1 = int(0.8 * len(data_shuffled))
split_2 = int(0.9 * len(data_shuffled))
splits = [
    (urls_merged[:split_1], VAMPIRE_DIR / 'bias-train.jsonl'),
    (urls_merged[split_1:split_2], VAMPIRE_DIR / 'bias-dev.jsonl'),
    (urls_merged[split_2:], VAMPIRE_DIR / 'bias-test.jsonl')
]

for df, path in splits:
    tqdm.write(str(path) + '\n')
    with path.open('a') as f:
        write_data_file(df, f)
