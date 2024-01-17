import io
import os
import json
import gzip
import zstandard as zstd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict

DCTX = zstd.ZstdDecompressor(max_window_size=2**31)

DATA_DIR = "/gscratch/cse/stelli/KnowledgePropogation/data"
PILE_DOMAINS = ['OpenWebText2', 'PubMed Abstracts', 'Github', 'StackExchange', 'Enron Emails', 'FreeLaw', 'USPTO Backgrounds', 'Pile-CC', 'Wikipedia (en)', 'Books3', 'PubMed Central', 'HackerNews', 'Gutenberg (PG-19)', 'DM Mathematics', 'NIH ExPorter', 'ArXiv', 'BookCorpus2', 'OpenSubtitles', 'YoutubeSubtitles', 'Ubuntu IRC', 'EuroParl', 'PhilPapers']
# domains_we_care_about = ['PubMed Abstracts', 'Wikipedia (en)']

def load_pile(split, subset, jsonl_path=None, train_limit=1000000000):
    data = []
    if jsonl_path is not None:
        if subset in PILE_DOMAINS:
            # extract the domain data
            with gzip.open(jsonl_path) as f:
                for line in f:
                    dp = json.loads(line)
                    if dp['meta']['pile_set_name'] == subset:
                        data.append(dp["text"].strip())
        else:
            with open(jsonl_path) as f:
                for line in f:
                    dp = json.loads(line)
                    data.append(dp["text"].strip())
    elif subset in ["amazon", "imdb", "subj", "cc-news", "new-imdb", "new-imdb-raw", "new-amazon", "MIMIC_III"]:
        with gzip.open(os.path.join(DATA_DIR, "{}/{}.jsonl".format(subset, split))) as f:
            for line in f:
                dp = json.loads(line)
                data.append(dp["text"].strip())
    elif split=="train":
        data = load_pile_train(subset, train_limit)
    else:
        fn = os.path.join(DATA_DIR, "the-pile/{}.json.gz".format(split))
        assert os.path.exists(fn), fn
        with gzip.open(fn) as f:
            for line in f:
                dp = json.loads(line)
                if subset==dp["meta"]["pile_set_name"].replace(" ", "_").replace("-", "_"):
                    data.append(dp["text"].strip())
    return data

def load_pile_train(subset, limit=1000000000):
    data = []
    base_dir = os.path.join(DATA_DIR, "the-pile/train-gz")
    n_tokens = []
    np.random.random(2023)
    for fn in sorted(os.listdir(base_dir)):
        if fn.endswith(".json.gz") and fn.split("-")[0]==subset:
            with gzip.open(os.path.join(base_dir, fn), "r") as f:
                for line in f:
                    dp = json.loads(line.decode())
                    data.append(dp["text"].strip())
                    n_tokens.append(len(dp["text"].strip().split()))

    n_tot_tokens = np.sum(n_tokens)

    if limit and n_tot_tokens > limit:
        # When the data is too large, it doesn't fit into RAM during tokenization
        np.random.seed(2023)
        indices = np.random.permutation(range(len(data)))
        new_data = []
        tot = 0

        for i in indices:
            new_data.append(data[i])
            tot += n_tokens[i]
            if tot >= limit:
                break

        print ("Sampled %.2fM->%.2fM sequences (%dM->%dM tokens) for %s" % (
                len(data)/1000000,
                len(new_data)/1000000,
                n_tot_tokens/1000000,
                tot/1000000,
                subset))
        data = new_data
    else:
        print ("Load %.2fM sequences (%dM tokens) for %s" % (len(data)/1000000, n_tot_tokens/1000000, subset))

    return data

def read_lines_from_zst_file(zstd_file_path:Path):
    with zstd.open(zstd_file_path, mode='rb', dctx=DCTX) as zfh:
        with io.TextIOWrapper(zfh) as iofh:
            for line in iofh:
                yield line


if __name__=='__main__': # for debugging
    # data = load_pile('val', 'Wikipedia_(en)')
    # with open(os.path.join(DATA_DIR, 'extracted_pile', 'pile_val_wikipedia.txt'), 'w+') as f:
    #     for line in data:
    #         f.write(line + '\n')
    
    data = load_pile('val', 'PubMed_Abstracts')
    with open(os.path.join(DATA_DIR, 'extracted_pile', 'pile_val_pubmedabstract.txt'), 'w+') as f:
        for line in data:
            f.write(line + '\n')