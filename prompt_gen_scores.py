from utils.constants import PERSPECTIVE_API_KEY
from scripts.perspective_api_request import request
from scripts.create_db import unpack_scores
import json
import pickle

with open('output/prompt-gens-gpt2-og.pkl', 'rb') as f:
        gens = pickle.load(f)

prompts, gen_conts = zip(*gens)

with open('prompt_gen_responses.jsonl', 'a') as f:
    for response in request(gen_conts, api_key=PERSPECTIVE_API_KEY, requests_per_second=25, should_yield=True):
        print(json.dumps(response) if response else '', file=f)
