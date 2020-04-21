from utils.constants import OUTPUT_DIR
import pandas as pd
from itertools import chain

from scripts.perspective_api_request import perspective_api_request

GENERATION_LIMIT = 25

prompts_dir = OUTPUT_DIR / 'prompts'

df = pd.read_pickle(prompts_dir / 'prompts_n_50percent.pkl')
generations = df.generation.apply(lambda g: g[:25])
generations = list(chain.from_iterable(generations))
print(len(generations))

out_file = prompts_dir / 'prompts_n_50percent_perspective.jsonl'
perspective_api_request(generations, out_file, should_continue=True)
