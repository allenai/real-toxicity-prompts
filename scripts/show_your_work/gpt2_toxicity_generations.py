import expected_max_performance
import pandas as pd
from plot import one_plot
from tqdm import tqdm
import numpy as np


tqdm.pandas()


if __name__ == '__main__':
    print('reading data...')
    df = pd.read_json("prompts_n_50percent_gpt2.jsonl", lines=True)

    def remove_none(x):
        return [item for item in x if item is not None]

    def pad(x, length=25):
        while len(x) < length:
            x.append(x[-1])
        return x


    toxicities = df.generations.progress_apply(lambda x: [y['toxicity'] for y in x])

    toxicities = toxicities.progress_apply(remove_none)
    toxicities = toxicities[toxicities.apply(len) > 0]
    toxicities = toxicities.progress_apply(pad)

    samplemaxes = toxicities.progress_apply(expected_max_performance.samplemax)

    mean_samplemax = np.array(samplemaxes.apply(lambda x: x['mean']).tolist())
    one_plot(mean_samplemax, "Experiment Name", logx=False, plot_errorbar=False, avg_time=0)