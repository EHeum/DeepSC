"""Training (Algorithms 1-4) and evaluation loops for DeepSC."""
import math
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .channels import apply_channel
from .data import SentenceDataset
from .metrics import bleu_scores
from .models import MINE, DeepSC, complex_pairs, mi_lower_bound


def build_model(vocab, cfg):
    return DeepSC(len(vocab), vocab.pad, d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 8),
                  n_layers=cfg.get("n_layers", 3), d_ff=cfg.get("d_ff", 512),
                  symbols_per_word=cfg.get("symbols_per_word", 8), dropout=cfg.get("dropout", 0.1),
                  rx_positional=cfg.get("rx_positional", True), norm_first=cfg.get("norm_first", False))


def save_checkpoint(path, model, mine, cfg, history):
    torch.save({"model": model.state_dict(), "mine": mine.state_dict(), "cfg": cfg, "history": history}, path)


def load_checkpoint(path, vocab, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    model = build_model(vocab, ck["cfg"])
    model.load_state_dict(ck["model"])
    mine = MINE()
    mine.load_state_dict(ck["mine"])
    return model, mine, ck


def ce_loss(logits, tgt_out, pad):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=pad)


def set_trainable(model, freeze):
    """freeze: 'none' | 'semantic' (Fig. 5(a), new channel) | 'channel' (Fig. 5(b), new knowledge)."""
    for p in model.parameters():
        p.requires_grad_(True)
    if freeze == "semantic":
        for p in model.semantic_parameters():
            p.requires_grad_(False)
    elif freeze == "channel":
        for p in model.channel_parameters():
            p.requires_grad_(False)


def train(model, mine, train_sents, test_sents, vocab, cfg, log=print):
    """Algorithm 1: alternate MI-model training (Alg. 2) and whole-network training (Alg. 3).

    cfg keys: channel, snr_min, snr_max, epochs, batch_size, lr, mine_lr, mi_lambda,
              freeze, unfreeze_epoch, warmup_steps, channel_kwargs
    """
    dev = next(model.parameters()).device
    ds = SentenceDataset(train_sents, vocab)
    dl = DataLoader(ds, cfg["batch_size"], shuffle=True, collate_fn=ds.collate, drop_last=True)
    test_ds = SentenceDataset(test_sents, vocab)
    test_dl = DataLoader(test_ds, 256, collate_fn=test_ds.collate)
    lam, ch, kw = cfg["mi_lambda"], cfg["channel"], cfg.get("channel_kwargs", {})
    set_trainable(model, cfg.get("freeze", "none"))

    def make_opt():
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-8)

    opt = make_opt()
    opt_mine = torch.optim.Adam(mine.parameters(), lr=cfg["mine_lr"])
    warm = cfg.get("warmup_steps", 0)
    step, history = 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        if cfg.get("unfreeze_epoch") and epoch == cfg["unfreeze_epoch"]:
            set_trainable(model, "none")  # Alg. 4: finally fine-tune the whole network
            opt = make_opt()
        model.train()
        t0, tot_ce, tot_mi, n = time.time(), 0.0, 0.0, 0
        for src in dl:
            src = src.to(dev)
            snr = random.uniform(cfg["snr_min"], cfg["snr_max"])
            # ---- phase 1: train the MI estimation model T (Alg. 2) ----
            with torch.no_grad():
                x, mask = model.transmit(src)
                y = apply_channel(ch, x, snr, **kw) * mask
            xs, ys = complex_pairs(x, y, mask)
            lb = mi_lower_bound(mine, xs, ys)
            opt_mine.zero_grad()
            (-lb).backward()
            opt_mine.step()
            # ---- phase 2: train the whole network with L_total (eq. (14), Alg. 3) ----
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * min(1.0, (step + 1) / warm) if warm else cfg["lr"]
            logits, x, y, mask = model(src, src[:, :-1], ch, snr, **kw)
            loss = ce = ce_loss(logits, src[:, 1:], vocab.pad)
            if lam > 0:
                xs, ys = complex_pairs(x, y, mask)
                loss = ce - lam * mi_lower_bound(mine, xs, ys)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tot_ce += ce.item()
            tot_mi += lb.item()
            n += 1
        test_ce = evaluate_ce(model, test_dl, vocab, ch, cfg.get("eval_snr", 6.0), **kw)
        rec = {"epoch": epoch, "train_ce": tot_ce / n, "mi_bits": tot_mi / n / math.log(2),
               "test_ce": test_ce, "sec": time.time() - t0}
        history.append(rec)
        log(f"epoch {epoch:3d} | train CE {rec['train_ce']:.4f} | test CE@{cfg.get('eval_snr', 6.0)}dB "
            f"{test_ce:.4f} | MI {rec['mi_bits']:.3f} bit/sym | {rec['sec']:.0f}s")
    return history


@torch.no_grad()
def evaluate_ce(model, dl, vocab, channel, snr, **kw):
    model.eval()
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for src in dl:
        src = src.to(dev)
        logits, *_ = model(src, src[:, :-1], channel, snr, **kw)
        tot += ce_loss(logits, src[:, 1:], vocab.pad).item() * src.size(0)
        n += src.size(0)
    return tot / n


@torch.no_grad()
def transmit_sentences(model, sentences, vocab, channel, snr, batch_size=256, **kw):
    """Send sentences through DeepSC and return the recovered sentences."""
    model.eval()
    dev = next(model.parameters()).device
    ds = SentenceDataset(sentences, vocab)
    out = []
    for i in range(0, len(ds), batch_size):
        src = ds.collate([ds[j] for j in range(i, min(i + batch_size, len(ds)))]).to(dev)
        dec = model.greedy_decode(src, vocab.start, vocab.end, channel, snr, **kw)
        out.extend(vocab.decode(d) for d in dec.tolist())
    return out


def evaluate_bleu(model, sentences, vocab, channel, snrs, **kw):
    refs = [vocab.decode(vocab.encode(s)) for s in sentences]  # what can be represented
    res = {}
    for snr in snrs:
        hyp = transmit_sentences(model, sentences, vocab, channel, snr, **kw)
        res[snr] = {"bleu": bleu_scores(refs, hyp), "samples": list(zip(refs[:5], hyp[:5]))}
    return res
