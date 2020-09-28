import click
from pathlib import Path

import pandas as pd
from hashlib import md5
import tldextract
from tqdm.auto import tqdm


def get_domain(url):
    ext = tldextract.extract(url)

    # Handle wordpress blogs (there are many)
    if ext.domain == 'wordpress':
        return '.'.join(ext)
    else:
        return '.'.join(ext[1:])


@click.command(help='Outputs CSV file containing URLs, MD5-hashed URLs, and domains')
@click.option('--urls-dir', required=True,
              help='Unarchived URLs directory containing files with newline-separated URLs')
@click.argument('output-file')
def process_urls(urls_dir: str, output_file: str):
    urls_dir = Path(urls_dir)
    urls_subsets = list(urls_dir.iterdir())

    rows = []
    for file in tqdm(urls_subsets, desc='Processing URL subsets'):
        with open(file) as urls:
            for url in urls:
                url = url.strip()
                hashed_url = md5(url.encode()).hexdigest()
                rows.append((url, hashed_url))

    print("Extracting domains...")
    df = pd.DataFrame(rows, columns=['url', 'md5_hash'])
    df['domain'] = df['url'].apply(get_domain)

    print("Writing CSV...")
    df.to_csv(output_file, index=False)

    print("Done!")


if __name__ == '__main__':
    process_urls()
