import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm, trange
import seaborn as sns
import json
import codecs


sns.set(context='paper', style='white')

tqdm.pandas()


def _cdf_with_replacement(i,n,N):
    return (i/N)**n

def _compute_variance(N, cur_data, expected_max_cond_n, pdfs):
    """
    this computes the standard error of the max.
    this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
    is stated in the last sentence of the summary on page 10 of 
    http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
    """
    variance_of_max_cond_n = []
    for n in range(N):
        # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
        cur_var = 0
        for i in range(N):
            cur_var += (cur_data[i] - expected_max_cond_n[n])**2 * pdfs[n][i]
        cur_var = np.sqrt(cur_var)
        variance_of_max_cond_n.append(cur_var)
    return variance_of_max_cond_n
    

# this implementation assumes sampling with replacement for computing the empirical cdf
def samplemax(validation_performance):
    validation_performance = list(validation_performance)
    validation_performance.sort()
    N = len(validation_performance)
    pdfs = []
    for n in range(1,N+1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1,N+1):
            F_Y_of_y.append(_cdf_with_replacement(i,n,N))


        f_Y_of_y = []
        cur_cdf_val = 0
        for i in range(len(F_Y_of_y)):
            f_Y_of_y.append(F_Y_of_y[i] - cur_cdf_val)
            cur_cdf_val = F_Y_of_y[i]
        
        pdfs.append(f_Y_of_y)

    expected_max_cond_n = []
    for n in range(N):
        # for a given n, estimate expected value with \sum(x * p(x)), where p(x) is prob x is max.
        cur_expected = 0
        for i in range(N):
            cur_expected += validation_performance[i] * pdfs[n][i]
        expected_max_cond_n.append(cur_expected)


    var_of_max_cond_n = _compute_variance(N, validation_performance, expected_max_cond_n, pdfs)

    return {"mean":expected_max_cond_n, "var":var_of_max_cond_n, "max": np.max(validation_performance),
            "min":np.min(validation_performance)}

def remove_none(x):
    return [y for y in x if y is not None and not np.isnan(y)]

def pad(x, length=25):
    while len(x) < length:
        x.append(x[-1])
    return x

def save_plot(data_name, logx, plot_errorbar, avg_time):
    name = "plots/{}_logx={}_errorbar={}_avgtime={}.pdf".format(data_name, logx, plot_errorbar, avg_time)

    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.savefig(name, dpi=300)


def one_plot(datas, data_name, models_to_plot, palette, logx=False, plot_errorbar=True, avg_time=0, performance_metric="accuracy", output_file="out.pdf"):
    # to set default values
    linestyle = "-"
    linewidth = 2
    errorbar_kind = 'shade'
    errorbar_alpha = 0.1
    fontsize = 16
    x_axis_time = avg_time != 0
    _, cur_ax = plt.subplots(1,1, fig)

    cur_ax.set_title(data_name, fontsize=fontsize)
    cur_ax.set_ylabel("Expected max toxicity", fontsize=fontsize)

    if x_axis_time:
        cur_ax.set_xlabel("Training duration",fontsize=fontsize)
    else:
        cur_ax.set_xlabel("Number of Generations",fontsize=fontsize)

    if logx:
        cur_ax.set_xscale('log')

    for label, data in datas.items():

        means = data['mean']
        vars = data['var']
        max_acc = data['max']
        min_acc = data['min']

        if x_axis_time:
            x_axis = [avg_time * (i+1) for i in range(len(means))]
        else:
            x_axis = [i+1 for i in range(len(means))]

        if plot_errorbar:
            if errorbar_kind == 'shade':
                minus_vars = [x - y if (x - y) >= min_acc else min_acc for x,y in zip(means, vars)]
                plus_vars = [x + y if (x + y) <= max_acc else max_acc for x,y in zip(means, vars)]
                plt.fill_between(x_axis,
                                minus_vars,
                                plus_vars,
                                alpha=errorbar_alpha)
            else:
                cur_ax.errorbar(x_axis,
                                means,
                                yerr=vars,
                                linestyle=linestyle,
                                linewidth=linewidth)
        sns.lineplot(x_axis,
                     means,
                     ax=cur_ax,
                     label=label,
                     palette=palette,
                     linestyle=linestyle,
                     linewidth=linewidth)

        # left, right = cur_ax.get_xlim()

        # plt.xlim((left, right))
        # plt.locator_params(axis='y', nbins=10)
        plt.tight_layout()

    handles,labels = cur_ax.get_legend_handles_labels()
    legend_pairs = dict(zip(labels, handles))
    legend_ordered = [(x, legend_pairs[x]) for x in models_to_plot]
    handles = [x[1] for x in legend_ordered]
    labels = [x[0] for x in legend_ordered]
    cur_ax.legend(handles,labels)
        
    if not os.path.exists("plots"):
        os.mkdir("plots")
    plt.show()


def get_samplemaxes_eos(df, label, prompt_condition=None):
    toxicities = pd.Series(df.toxicity.values.reshape(df.shape[0] // 25, 25).tolist())
    toxicities = toxicities.apply(remove_none)
    toxicities = toxicities[toxicities.apply(len) > 0]
    toxicities = toxicities.apply(pad)
    samplemaxes = toxicities.apply(samplemax)
    mean = np.array(samplemaxes.apply(lambda x: x['mean']).tolist())
    var = np.array(samplemaxes.apply(lambda x: x['var']).tolist())
    max_ = np.array(samplemaxes.apply(lambda x: x['max']).tolist())
    min_ = np.array(samplemaxes.apply(lambda x: x['min']).tolist())
    return {"mean": mean.mean(0), "var": var.mean(0), "max": max_.mean(), "min": min_.mean()}

def get_samplemaxes(df, label, prompt_condition=None):
    
    df = df.loc[df.prompt.apply(lambda x: (x['toxicity'] is not None))]  
    if prompt_condition:
        df = df.loc[df.prompt.apply(prompt_condition)]
        if df.shape[0] == 0:
            raise Exception(f"prompt condition at {label}['toxicity_condition'] returns 0 elements")
    toxicities = df.generations.apply(lambda x: [y['toxicity'] for y in x ])
    toxicities = toxicities.apply(remove_none)
    toxicities = toxicities[toxicities.apply(len) > 0]
    toxicities = toxicities.apply(pad)
    samplemaxes = toxicities.apply(samplemax)
    mean = np.array(samplemaxes.apply(lambda x: x['mean']).tolist())
    var = np.array(samplemaxes.apply(lambda x: x['var']).tolist())
    max_ = np.array(samplemaxes.apply(lambda x: x['max']).tolist())
    min_ = np.array(samplemaxes.apply(lambda x: x['min']).tolist())
    return {"mean": mean.mean(0), "var": var.mean(0), "max": max_.mean(), "min": min_.mean()}


def read_jsonl(input_filename):
    data = []
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        for line in tqdm(input_file):
            data.append(json.loads(line, encoding='utf-8'))
    return data


if __name__ == '__main__':
    print('reading data...')


    models_to_plot = ['DAPT (Non-Toxic)', 'CTRL Tokens', 'GPT-2', 'DAPT (Toxic)']

    models = {
        "GPT-2": {
            "path":  "../data/prompt_n_50percent_toxicity_quartiles_25k_subsample/prompt_n_50percent_toxicity_quartiles_25k_subsample_gpt2.jsonl",
        },
        "DAPT (Toxic)": {
            "path":  "../data/prompt_n_50percent_toxicity_quartiles_25k_subsample/prompt_n_50percent_toxicity_quartiles_25k_subsample_gpt2_ft_gte99.jsonl",
        },
        "DAPT (Non-Toxic)": {
            "path":  "../data/prompt_n_50percent_toxicity_quartiles_25k_subsample/prompt_n_50percent_toxicity_quartiles_25k_subsample_gpt2_ft_lte2.jsonl",
        },
        "CTRL Tokens": {
            "path":  "../data/prompt_n_50percent_toxicity_quartiles_25k_subsample/prompt_n_50percent_toxicity_quartiles_25k_subsample_gpt2_ctrl.jsonl",
        },
         "Affect-GPT2": {
            "path":  "../data/prompt_n_50percent_toxicity_quartiles_25k_subsample/prompt_n_50percent_toxicity_quartiles_25k_subsample_affect_beta_3.jsonl",
        },
         "PPLM": {
            "path":  "../data/prompt_n_50percent_toxicity_quartiles_25k_subsample/prompt_n_50percent_toxicity_quartiles_25k_subsample_pplm_10k_prompt_10_samples.jsonl",
        }
    }

    dfs = []
    for model in models_to_plot:
        subdf = pd.DataFrame(read_jsonl(models[model]['path']))
        subdf['model'] = model
        dfs.append(subdf)

    df = pd.concat(dfs, 0)
    palette = sns.color_palette("coolwarm", len(models))
    prompt_conditions = {
        "GPT-2": 
            {
                "general_condition": df.model == 'GPT-2',
                "toxicity_condition": lambda x: x['toxicity'] > 0.5,
            },
        "DAPT (Non-Toxic)": {
                "general_condition": df.model == 'DAPT (Non-Toxic)',
                "toxicity_condition": lambda x: x['toxicity'] > 0.5,
            },
        "DAPT (Toxic)": {
                "general_condition": df.model == 'DAPT (Toxic)',
                "toxicity_condition": lambda x: x['toxicity'] > 0.5,
        },
        "CTRL Tokens": {
                "general_condition": df.model == 'CTRL Tokens',
                "toxicity_condition": lambda x: x['toxicity'] > 0.5,
        },
        "Affect-GPT2": {
                "general_condition": df.model == 'Affect-GPT2',
                "toxicity_condition": lambda x: x['toxicity'] > 0.5,
        },
        "PPLM": {
                "general_condition": df.model == 'PPLM',
                "toxicity_condition": lambda x: x['toxicity'] > 0.5,
        }
    }
    
    dfs = []
    for model in models_to_plot:
        if prompt_conditions[model].get('general_condition') is not None:
            subdf = df.loc[prompt_conditions[model]['general_condition']]
        else:
            subdf = df
        if subdf.shape[0] == 0:
            raise Exception(f"prompt condition at {model}['general_condition'] returns 0 elements")
        dfs.append({"df": subdf,
                    "label": model,
                    "toxicity_condition": prompt_conditions[model]['toxicity_condition']})
    samplemaxes = []
    for df_ in tqdm(dfs):
        samplemax_ = get_samplemaxes(df_['df'], df_['label'], df_['toxicity_condition'])
        samplemaxes.append({df_['label']: samplemax_})
    print("plotting...")
    samplemaxes_merge = {k: v for d in samplemaxes for k, v in d.items()}
    one_plot(samplemaxes_merge, 
             "",
             logx=False,
             plot_errorbar=False,
             avg_time=0,
             palette=palette,
             models_to_plot=models_to_plot,
             output_file="data_tweaks_toxic_prompts.pdf")
    print("Done!")