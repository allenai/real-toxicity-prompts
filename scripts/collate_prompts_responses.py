# %%
import json
from pathlib import Path
import pandas as pd

from scripts.create_db import unpack_scores
from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.utils import batchify
from tqdm import tqdm

# %%
archive = Path('../archive')
prompts = archive / 'prompts' / 'promptsv2' / 'prompts_n_50percent'


# %%
def load_scores(scores_file: Path):
    scores = []
    for line in tqdm(open(scores_file)):
        response = json.loads(line)
        if response:
            summary_scores, _ = unpack_scores(response)
        else:
            summary_scores = None
        scores.append(summary_scores)
    return scores


def load_scores_2(scores_file: Path):
    scores = []
    for line in tqdm(open(scores_file)):
        response = json.loads(line)
        if response['response']:
            summary_scores, _ = unpack_scores(response['response'])
        else:
            summary_scores = None
        scores.append(summary_scores)
    return scores


# %%
df = pd.read_pickle('/data/language-model-toxicity/results/prompts/prompts_n_50percent_gpt2.pkl')
og_df = pd.read_pickle(prompts / 'dataset.pkl')
prompt_fix_df = pd.read_pickle(prompts / 'prompt_fix' / 'dataset.pkl')
cont_fix_df = pd.read_pickle(prompts / 'cont_fix' / 'dataset.pkl')

# %%
prompt_scores = load_scores(prompts / 'prompts.jsonl')
cont_scores = load_scores(prompts / 'continuations.jsonl')
prompt_fix_scores = load_scores_2(prompts / 'prompt_fix' / 'responses.jsonl')
cont_fix_scores = load_scores_2(prompts / 'cont_fix' / 'responses.jsonl')

# %%
og_df['prompt_responses'] = prompt_scores
og_df['cont_responses'] = cont_scores
prompt_fix_df['prompt_responses'] = prompt_fix_scores
cont_fix_df['cont_responses'] = cont_fix_scores

# %%
prompt_responses = og_df['prompt_responses'].combine_first(prompt_fix_df['prompt_responses'])
cont_responses = og_df['cont_responses'].combine_first(cont_fix_df['cont_responses'])

# %%
gen_file = Path(
    archive / 'prompts/promptsv2/prompts_n_50percent_temp/prompts_n_50percent_gpt2_generations_perspective.jsonl'
)
gen_responses = load_scores_2(gen_file)


# %%
def format_subtable(response, text):
    if not response:
        response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
    return {'text': text, **response}


# %%
with open('prompts-n50p-out.jsonl', 'a') as f:
    for (i, row), prompt_response, cont_response, gen_response in tqdm(zip(df.iterrows(),
                                                                           prompt_responses,
                                                                           cont_responses,
                                                                           batchify(gen_responses, 25)),
                                                                       total=len(df)):
        out = {
            'filename': row.filename,
            'begin': row.begin,
            'end': row.end,
            'prompt': format_subtable(prompt_response, row.prompt),
            'continuation': format_subtable(cont_response, row.continuation),
            'generations': [format_subtable(scores, gen) for gen, scores in zip(row.generation, gen_response)]
        }
        print(json.dumps(out), file=f)

# %%
df = pd.read_json('prompts-n50p-out.jsonl', orient='records')

