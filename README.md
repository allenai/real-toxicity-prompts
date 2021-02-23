# Real Toxicity Prompts
[[Paper link]](https://arxiv.org/abs/2009.11462)
[[Demo link]](https://toxicdegeneration.allenai.org/) 
[[Data link]](https://allenai.org/data/real-toxicity-prompts)

Work in progress, please revisit this page soon for more in-depth instructions.

## Installing
Run the commands provided in order to set up your environment:
1. `git clone https://github.com/allenai/real-toxicity-prompts.git`
1. `cd real-toxicity-prompts`
2. Install [Anaconda](https://docs.anaconda.com/anaconda/install/), then run `conda env create -f environment.yml` to create a new Anaconda environment named `rtp` 
with all necessary dependencies.
4. Run `conda activate rtp` before running any further scripts or notebooks.

## Getting Started
### Getting the data
Data processing steps are forthcoming.

### Getting API access
We used the following two APIs to run our experiments:
1. [Perspective API](https://github.com/conversationai/perspectiveapi/tree/master/1-get-started) to score text for toxicity
2. [OpenAI API](https://beta.openai.com/) to generate text with the GPT-3 model

### Reproducing the experiments
The prompts experiments can be run from `scripts/run_prompts_experiments.py`. 

Type `python -m scripts.run_prompts_experiments --help` to get a list of the available models and arguments that can 
be used.
