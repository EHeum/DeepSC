"""Europarl text preprocessing, vocabulary and PyTorch dataset.

Follows Sec. V-A of the paper: sentences of the European Parliament
proceedings are normalised and only sentences with 4-30 words are kept.
"""
import json
import random
import re
import unicodedata
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD, START, END, UNK = "<PAD>", "<START>", "<END>", "<UNK>"
SPECIALS = [PAD, START, END, UNK]

# Character set used by the text normaliser: 26 letters, space and 5
# punctuation marks = 32 symbols, i.e. exactly 5 bits (the "5-bit" baseline).
CHARSET = "abcdefghijklmnopqrstuvwxyz .,?!'"


def normalize(line: str) -> str:
    s = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"<[^>]*>", " ", s)  # xml tags
    s = re.sub(r"\([^)]*\)", " ", s)  # stage directions, e.g. "( The House rose )"
    # Europarl splits contractions ("minute ' s"); glue them back together
    s = re.sub(r"\s*'\s*(s|t|re|ve|ll|d|m)\b", r"'\1", s)
    s = re.sub(r"(^|\s)'|'(\s|$)", " ", s)  # quote marks
    s = re.sub(r"[^a-z.,!?' ]", " ", s)
    s = re.sub(r"([.,!?])", r" \1 ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_sentences(paths, min_len=4, max_len=30, seed=0):
    sents, seen = [], set()
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = normalize(line)
                n = len(s.split())
                if min_len <= n <= max_len and s not in seen:
                    seen.add(s)
                    sents.append(s)
    random.Random(seed).shuffle(sents)
    return sents


class Vocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, sentences, min_freq=1):
        cnt = Counter(w for s in sentences for w in s.split())
        words = sorted((w for w, c in cnt.items() if c >= min_freq), key=lambda w: (-cnt[w], w))
        return cls(SPECIALS + words)

    def __len__(self):
        return len(self.itos)

    @property
    def pad(self):
        return self.stoi[PAD]

    @property
    def start(self):
        return self.stoi[START]

    @property
    def end(self):
        return self.stoi[END]

    def encode(self, sentence):
        unk = self.stoi[UNK]
        return [self.start] + [self.stoi.get(w, unk) for w in sentence.split()] + [self.end]

    def decode(self, ids):
        words = []
        for i in ids:
            i = int(i)
            if i == self.end:
                break
            if i in (self.pad, self.start):
                continue
            words.append(self.itos[i])
        return " ".join(words)


def save_corpus(path, train, test, vocab):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"train": train, "test": test, "vocab": vocab.itos}, f)


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["train"], d["test"], Vocab(d["vocab"])


class SentenceDataset(Dataset):
    def __init__(self, sentences, vocab):
        self.sentences = sentences
        self.ids = [vocab.encode(s) for s in sentences]
        self.pad = vocab.pad

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return torch.tensor(self.ids[i], dtype=torch.long)

    def collate(self, batch):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=self.pad)
