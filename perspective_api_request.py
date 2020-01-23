from googleapiclient import discovery
import json
import time
from tqdm import tqdm
from typing import List

from constants import TEXTS_DIR, PERSPECTIVE_API_RESPONSE_DIR, PERSPECTIVE_API_FAILURES_FILE, \
    PERSPECTIVE_API_LENGTH_LIMIT_FAILURE_FILE, PERSPECTIVE_API_KEY, PERSPECTIVE_API_LEN_LIMIT, \
    PERSPECTIVE_API_SLEEP_SECONDS, PERSPECTIVE_API_ATTRIBUTES

# Generates API client object dynamically based on service name and version.
SERVICE = discovery.build('commentanalyzer', 'v1alpha1', developerKey=PERSPECTIVE_API_KEY)
BATCH_SIZE = 25
REQUESTS = {}
NUM_FAILURES = 0
NUM_FAILURES_TOO_LONG = 0


def perspective_request(text: str):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES}
    }
    return SERVICE.comments().analyze(body=analyze_request)


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

    too_long_filenames = PERSPECTIVE_API_LENGTH_LIMIT_FAILURE_FILE.read_text().split()
    too_long_files = set(TEXTS_DIR / filename for filename in too_long_filenames)

    # Remove all failed downloads from pending files
    pending_files -= failure_text_files
    pending_files -= too_long_files

    return list(pending_files)


def chunk_text(text: str, chunk_len: int) -> List[str]:
    chunks = []
    for i in range(0, len(text), chunk_len):
        chunks.append(text[i:i + chunk_len])
    return chunks


def response_callback(request_id, response, exception):
    global NUM_FAILURES

    text_filename = request_id
    response_file = PERSPECTIVE_API_RESPONSE_DIR / (text_filename + ".json")
    if exception:
        with PERSPECTIVE_API_FAILURES_FILE.open('a') as f:
            print(text_filename, file=f)
        NUM_FAILURES += 1
    else:
        try:
            with response_file.open('w') as f:
                json.dump(response, f)
        except OSError as e:
            print(e)


def request_files(pending_files):
    # TODO: add chunking
    global NUM_FAILURES_TOO_LONG

    pbar = tqdm(total=len(pending_files))
    i = 0
    while i < len(pending_files):
        # Get items for batch
        batch_files = pending_files[i: i + BATCH_SIZE]

        # Request batch
        batch_request = SERVICE.new_batch_http_request()
        for file in batch_files:
            text = file.read_text()
            if len(text) > PERSPECTIVE_API_LEN_LIMIT:
                with PERSPECTIVE_API_LENGTH_LIMIT_FAILURE_FILE.open('a') as f:
                    print(file.name, file=f)
                NUM_FAILURES_TOO_LONG += 1
                continue
            request_id = file.name
            batch_request.add(perspective_request(text), callback=response_callback, request_id=request_id)

        before = time.time()
        batch_request.execute()
        after = time.time()

        # Update progress bar
        i += BATCH_SIZE
        pbar.update(BATCH_SIZE)
        pbar.set_description(f"too long: {NUM_FAILURES_TOO_LONG}, failures: {NUM_FAILURES}")

        # Sleep off remainder time
        request_time = after - before
        if request_time < PERSPECTIVE_API_SLEEP_SECONDS:
            time.sleep(PERSPECTIVE_API_SLEEP_SECONDS - request_time)


if __name__ == '__main__':
    tqdm.write("Finding pending files...\n")
    pending_files = find_pending_files()
    tqdm.write("Requesting from Perspective API...\n")
    request_files(pending_files)
