"""Train DeepSC (Algorithm 1) or re-train a pre-trained one with transfer learning (Algorithm 4)."""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepsc.data import load_corpus  # noqa: E402
from deepsc.engine import build_model, save_checkpoint, train  # noqa: E402
from deepsc.models import MINE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/europarl_en.json")
    ap.add_argument("--channel", default="awgn", choices=["awgn", "rayleigh", "rician", "erasure"])
    ap.add_argument("--erasure-p", type=float, default=0.1)
    ap.add_argument("--rician-k", type=float, default=1.0)
    ap.add_argument("--snr-min", type=float, default=0.0)
    ap.add_argument("--snr-max", type=float, default=18.0)
    ap.add_argument("--eval-snr", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mine-lr", type=float, default=1e-3)
    ap.add_argument("--mine-out", default="linear", choices=["linear", "relu"],
                    help="activation of the MI model's output layer (Table I: ReLU)")
    ap.add_argument("--mi-lambda", type=float, default=0.01, help="lambda of eq. (14); 0 = no MI term")
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--symbols-per-word", type=int, default=8)
    ap.add_argument("--no-rx-pos", action="store_true", help="do not add positional encoding at the receiver")
    ap.add_argument("--constellation", default=None,
                    help="digital channel input, e.g. qam4 / qam16 / qam64 / psk8 (default: analog)")
    ap.add_argument("--quant-temp", type=float, default=0.5, help="soft-quantiser temperature (units of d_min^2)")
    ap.add_argument("--quant-ste", default="soft", choices=["soft", "identity"],
                    help="backward pass of the quantiser: soft assignment (DeepJSCC-Q) or identity")
    ap.add_argument("--rx-hard", action="store_true", help="hard symbol decisions at the receiver")
    ap.add_argument("--init", default=None, help="pre-trained checkpoint (transfer learning)")
    ap.add_argument("--freeze", default="none", choices=["none", "semantic", "channel"])
    ap.add_argument("--unfreeze-epoch", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N training sentences")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    train_s, test_s, vocab = load_corpus(a.data)
    if a.max_train:
        train_s = train_s[: a.max_train]

    kw = {"erasure": {"p": a.erasure_p}, "rician": {"k": a.rician_k}}.get(a.channel, {})
    cfg = {"channel": a.channel, "channel_kwargs": kw, "snr_min": a.snr_min, "snr_max": a.snr_max,
           "eval_snr": a.eval_snr, "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
           "mine_lr": a.mine_lr, "mi_lambda": a.mi_lambda, "warmup_steps": a.warmup_steps,
           "symbols_per_word": a.symbols_per_word, "freeze": a.freeze, "unfreeze_epoch": a.unfreeze_epoch,
           "init": a.init, "seed": a.seed, "rx_positional": not a.no_rx_pos,
           "constellation": a.constellation, "quant_temp": a.quant_temp, "quant_ste": a.quant_ste,
           "rx_hard": a.rx_hard, "mine_out": a.mine_out}
    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        cfg["symbols_per_word"] = ck["cfg"]["symbols_per_word"]
        cfg["rx_positional"] = ck["cfg"].get("rx_positional", True)
        # the architecture may change (e.g. analog -> constellation); the weights are identical
        model, mine = build_model(vocab, cfg), MINE(out_act=a.mine_out)
        model.load_state_dict(ck["model"])
        mine.load_state_dict(ck["mine"])
    else:
        model, mine = build_model(vocab, cfg), MINE(out_act=a.mine_out)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mine = model.to(dev), mine.to(dev)
    print(json.dumps(cfg))
    print(f"vocab={len(vocab)} train={len(train_s)} test={len(test_s)} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    history = train(model, mine, train_s, test_s, vocab, cfg, log=lambda s: print(s, flush=True))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_checkpoint(a.out, model.cpu(), mine.cpu(), cfg, history)
    with open(os.path.splitext(a.out)[0] + "_history.json", "w") as f:
        json.dump({"cfg": cfg, "history": history}, f, indent=1)


if __name__ == "__main__":
    main()
