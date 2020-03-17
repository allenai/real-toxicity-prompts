import numpy as np
import torch

from utils.constants import TEXTS_DIR


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def load_text(filename: str) -> str:
    return (TEXTS_DIR / filename).read_text()
