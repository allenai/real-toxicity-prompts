from utils.constants import DATA_DIR
import pickle
from utils.generation import GPT2Generator
from tqdm.auto import trange

pkl = DATA_DIR / 'ngram-beginning-prompts.pkl'
df = pickle.load(pkl.open('rb'))
df = df.dropna()

generator = GPT2Generator()
batch_size = 8
prompts = df.prompt
gens = []

for i in trange(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i + batch_size]
    batch_gens = generator.generate(batch_prompts, max_len=50)
    gens.extend(list(zip(batch_prompts, batch_gens)))

with open('prompt-gens-gpt2-og.pkl', 'wb') as f:
    pickle.dump(gens, f)
