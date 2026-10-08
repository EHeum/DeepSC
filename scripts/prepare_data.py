"""Download (optional) and preprocess the Europarl English corpus.

The full Europarl v7 (2.0M sentences, statmt.org) is used in the paper. If it is
not reachable, the NLTK `europarl_raw` sample (10 sessions of the English
proceedings) mirrored on GitHub is used instead.
"""
import argparse
import glob
import io
import os
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepsc.data import Vocab, load_sentences, save_corpus  # noqa: E402

NLTK_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/europarl_raw.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="directory / glob of raw English Europarl text files")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/europarl_en.json")
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.raw:
        files = sorted(glob.glob(os.path.join(a.raw, "*")) if os.path.isdir(a.raw) else glob.glob(a.raw))
    else:
        files = sorted(glob.glob(os.path.join(a.raw_dir, "europarl_raw", "english", "*.en")))
        if not files:
            print("downloading", NLTK_URL)
            data = urllib.request.urlopen(NLTK_URL).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for n in z.namelist():
                    if "/english/" in n and n.endswith(".en"):
                        z.extract(n, a.raw_dir)
            files = sorted(glob.glob(os.path.join(a.raw_dir, "europarl_raw", "english", "*.en")))

    sents = load_sentences(files, seed=a.seed)
    n_test = int(len(sents) * a.test_ratio)
    test, train = sents[:n_test], sents[n_test:]
    vocab = Vocab.build(sents)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_corpus(a.out, train, test, vocab)
    lens = [len(s.split()) for s in sents]
    print(f"files={len(files)} sentences={len(sents)} train={len(train)} test={len(test)} "
          f"vocab={len(vocab)} avg_words={sum(lens) / len(lens):.2f}")


if __name__ == "__main__":
    main()
