from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
import plac
from spacy.lang.en import English
import json
from spacy.util import minibatch
from utils.utils import make_corpus_iter


@plac.annotations(
    corpus_dir=("Corpus directory", "positional", None, Path),
    output_dir=("Output directory", "positional", None, Path),
    n_jobs=("Number of workers", "option", "n", int),
    batch_size=("Batch-size for each process", "option", "b", int),
)
def main(corpus_dir, output_dir, n_jobs=4, batch_size=10_000):
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    if not output_dir.exists():
        output_dir.mkdir()

    print("Processing texts...")
    corpus_iter = make_corpus_iter(corpus_dir)
    partitions = minibatch(corpus_iter, size=batch_size)
    executor = Parallel(n_jobs=n_jobs, backend="multiprocessing", prefer="processes")
    do = delayed(partial(transform_texts, nlp))
    tasks = (do(i, batch, output_dir) for i, batch in enumerate(partitions))
    executor(tasks)


def transform_texts(nlp, batch_id, batch, output_dir):
    print(nlp.pipe_names)
    out_path = Path(output_dir) / (f"%d.jsonl" % batch_id)
    if out_path.exists():  # return None in case same batch is called again
        return None

    doc_ids, texts = zip(*batch)

    print("Processing batch", batch_id)
    with out_path.open("a", encoding="utf8") as f:
        for doc_id, doc in zip(doc_ids, nlp.pipe(texts)):
            json.dump({doc_id: [sentence_span.text for sentence_span in doc.sents]}, f)
            f.write("\n")
    print(f"Saved {len(texts)} texts to {out_path.name}")


if __name__ == "__main__":
    plac.call(main)
