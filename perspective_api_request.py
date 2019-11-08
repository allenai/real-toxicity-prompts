from googleapiclient import discovery
import json
import time
from tqdm import tqdm
from typing import List

from constants import TEXTS_DIR, PERSPECTIVE_API_RESPONSE_DIR, PERSPECTIVE_API_FAILURES_FILE, PERSPECTIVE_API_KEY

# Generates API client object dynamically based on service name and version.
SERVICE = discovery.build('commentanalyzer', 'v1alpha1', developerKey=PERSPECTIVE_API_KEY)

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/api_reference.md#toxicity-models
ATTRIBUTES = [
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
    'SEXUALLY_EXPLICIT',
    'FLIRTATION'
]

PERSPECTIVE_API_SLEEP_SECONDS = 1
PERSPECTIVE_API_LEN_LIMIT = 20000


def perspective_request(text):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {attr: {} for attr in ATTRIBUTES}
    }
    return SERVICE.comments().analyze(body=analyze_request).execute()


def response_file_for(text_file, chunk_num=None):
    if chunk_num is None:
        response_filename = text_file.name + '.json'
    else:
        response_filename = f'{text_file.name}.chunk-{chunk_num}.json'
    return PERSPECTIVE_API_RESPONSE_DIR / response_filename


def find_pending_files():
    pending_files = set()
    for text_file in TEXTS_DIR.iterdir():
        if not response_file_for(text_file).exists():
            pending_files.add(text_file)

    failure_text_filenames = PERSPECTIVE_API_FAILURES_FILE.read_text().split()
    failure_text_files = set(TEXTS_DIR / filename for filename in failure_text_filenames)

    # Remove all failed downloads from pending files
    pending_files -= failure_text_files

    return pending_files


def chunk_text(text: str, chunk_len: int) -> List[str]:
    chunks = []
    for i in range(0, len(text), chunk_len):
        chunks.append(text[i:i + chunk_len])
    return chunks


def request_files(pending_files):
    for text_file in tqdm(pending_files):
        full_text = text_file.read_text()
        chunks = chunk_text(full_text, PERSPECTIVE_API_LEN_LIMIT)

        for i, text in enumerate(chunks):
            if len(chunks) > 1:
                response_file = response_file_for(text_file, chunk_num=i)
            else:
                response_file = response_file_for(text_file)

            try:
                response = perspective_request(text)
                with response_file.open('w') as f:
                    json.dump(response, f)
            except:
                with PERSPECTIVE_API_FAILURES_FILE.open('a') as f:
                    print(text_file.name, file=f)

            # Sleep for 1 second due to rate limiting by API
            time.sleep(PERSPECTIVE_API_SLEEP_SECONDS)


def main():
    pending_files = find_pending_files()
    # pending_files = [TEXTS_DIR / '0466904-a2ca59bee16fc898dea2e3fae2a0b5ed.txt']
    request_files(pending_files)


main()
