# Project-level constants, including API keys and directories
# Note: importing this file has the side effect of loading a configuration file
from pathlib import Path
import yaml

##############################
# Config
##############################
CONFIG_FILE = Path('config.yml')
PERSPECTIVE_API_KEY = ''
OPENAI_API_KEY = ''
try:
    with open(CONFIG_FILE) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    PERSPECTIVE_API_KEY = config['perspective']
    OPENAI_API_KEY = config['openai']
except FileNotFoundError:
    print('No config file found. API keys will not be loaded.')

##############################
# Paths
##############################
SHARED_DIR = Path('/data_processing/language-model-toxicity')
DATA_DIR = SHARED_DIR / 'data_processing'
OUTPUT_DIR = Path('output')

OPENWEBTEXT_DB = DATA_DIR / 'openwebtext-perspective.db'
WEBTEXT_DB = DATA_DIR / 'webtext.db'

# TODO: remove
TEXTS_DIR = DATA_DIR / 'openwebtext'

##############################
# Perspective API
##############################
PERSPECTIVE_API_LEN_LIMIT = 20480

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
