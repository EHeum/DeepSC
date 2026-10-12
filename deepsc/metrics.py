"""Performance metrics of Sec. III-C: BLEU (eqs. (11)-(12)) and sentence similarity (eq. (13))."""
import math
from collections import Counter

import numpy as np


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu_ngram(reference, candidate, n):
    """Individual n-gram BLEU score (u_n = 1 for one n, 0 otherwise) of one sentence.

    log BLEU = min(1 - l_s / l_s_hat, 0) + log p_n   with clipped n-gram precision p_n.
    """
    ref, cand = reference.split(), candidate.split()
    c = _ngrams(cand, n)
    total = sum(c.values())
    if total == 0:
        return 0.0
    r = _ngrams(ref, n)
    p_n = sum(min(v, r[k]) for k, v in c.items()) / total
    bp = math.exp(min(1.0 - len(ref) / len(cand), 0.0))
    return bp * p_n


def bleu_scores(references, candidates, ns=(1, 2, 3, 4)):
    """Average individual n-gram BLEU over a corpus -> {n: score}."""
    return {n: float(np.mean([bleu_ngram(r, c, n) for r, c in zip(references, candidates)])) for n in ns}


class SentenceSimilarity:
    """match(s_hat, s) = B(s) . B(s_hat)^T / (||B(s)|| ||B(s_hat)||)   (eq. (13)).

    B is a pre-trained BERT; the sentence vector is the mean of the last hidden
    states. Requires `transformers` and a downloadable / local BERT checkpoint.
    """

    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        from transformers import AutoModel, AutoTokenizer  # optional dependency
        import torch

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    def embed(self, sentences, batch_size=64):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                enc = self.tok(sentences[i:i + batch_size], padding=True, truncation=True,
                               return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1)).cpu())
        return self.torch.cat(out)

    def __call__(self, references, candidates):
        a, b = self.embed(references), self.embed(candidates)
        sim = self.torch.nn.functional.cosine_similarity(a, b, dim=-1)
        return float(sim.mean())
