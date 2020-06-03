import re
from pathlib import Path

from tqdm import tqdm

from utils.utils import load_jsonl, first

dups_path = Path('/data/language-model-toxicity/data/lsh_duplicates')

prog = re.compile(r'webtext_([0-9]+)-([0-9]+)')
fix_id = lambda match: '-'.join(map(str, map(int, match.groups())))

with open(dups_path / 'duplicates_90.csv', 'w') as f:
    f.write('openwebtext_id,webtext_id')
    f.write('\n')
    for line in tqdm(load_jsonl(dups_path / 'duplicates_90.jsonl'), total=2441952):
        owtc_id, wt_ids = first(line.items())
        for wt_id in map(fix_id, map(prog.match, wt_ids)):
            f.write(f'{owtc_id},{wt_id}')
            f.write('\n')
