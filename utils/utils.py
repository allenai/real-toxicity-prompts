from utils.constants import TEXTS_DIR


def load_text(filename: str) -> str:
    return (TEXTS_DIR / filename).read_text()
