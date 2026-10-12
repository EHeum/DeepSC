import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepsc.baselines import (FixedLengthCoder, HuffmanCoder, QAM, ReedSolomon,  # noqa: E402
                              TraditionalSystem, TurboCode)
from deepsc.channels import apply_channel, power_normalize, snr_to_noise_std  # noqa: E402
from deepsc.data import SentenceDataset, Vocab, normalize  # noqa: E402
from deepsc.metrics import bleu_ngram  # noqa: E402
from deepsc.models import MINE, DeepSC, complex_pairs, mi_lower_bound  # noqa: E402

SENTS = ["the european parliament adopted the resolution .",
         "i would like to thank the commissioner for his reply .",
         "we must protect the environment in all member states ."]


def test_normalize():
    s = "( The House rose ) Madam President , I declare the minute ' s silence ."
    assert normalize(s) == "madam president , i declare the minute's silence ."


def test_model_shapes_and_power():
    v = Vocab.build(SENTS)
    ds = SentenceDataset(SENTS, v)
    src = ds.collate([ds[i] for i in range(3)])
    m = DeepSC(len(v), v.pad, symbols_per_word=8)
    logits, x, y, mask = m(src, src[:, :-1], "awgn", 10.0)
    assert logits.shape == (3, src.size(1) - 1, len(v))
    assert x.shape == (3, src.size(1) * 8) and x.is_complex()
    p = (x.abs() ** 2 * mask).sum(1) / mask.sum(1)  # unit power per sentence, pads not sent
    assert torch.allclose(p, torch.ones(3), atol=1e-4)
    assert (x[~mask] == 0).all()
    out = m.greedy_decode(src, v.start, v.end, "rayleigh", 10.0)
    assert out.size(0) == 3


def test_channel_snr():
    torch.manual_seed(0)
    x = power_normalize(torch.complex(torch.randn(64, 4096), torch.randn(64, 4096)))
    for snr in (0.0, 10.0):
        n = apply_channel("awgn", x, snr) - x
        measured = 10 * math.log10(1 / (n.abs() ** 2).mean().item())
        assert abs(measured - snr) < 0.1
    assert abs(snr_to_noise_std(0.0) - 1 / math.sqrt(2)) < 1e-9


def test_mine_learns_positive_mi():
    torch.manual_seed(0)
    mine = MINE()
    opt = torch.optim.Adam(mine.parameters(), 1e-3)
    for _ in range(300):
        x = power_normalize(torch.complex(torch.randn(1, 4096), torch.randn(1, 4096)))
        y = apply_channel("awgn", x, 10.0)
        xs, ys = complex_pairs(x, y, torch.ones_like(x.real, dtype=torch.bool))
        lb = mi_lower_bound(mine, xs, ys)
        opt.zero_grad()
        (-lb).backward()
        opt.step()
    bits = lb.item() / math.log(2)
    assert 1.5 < bits < math.log2(11) + 0.3  # Gaussian capacity at 10 dB = 3.46 bit


def test_bleu():
    assert bleu_ngram("a b c d", "a b c d", 1) == 1.0
    assert bleu_ngram("a b c d", "a b c d", 4) == 1.0
    assert bleu_ngram("a b c d", "x y z w", 1) == 0.0
    assert abs(bleu_ngram("a b c d", "a b", 1) - math.exp(1 - 2)) < 1e-9  # brevity penalty


def test_source_coders_roundtrip():
    for c in (HuffmanCoder(SENTS), FixedLengthCoder()):
        for s in SENTS:
            assert c.decode(c.encode(s)) == s


def test_qam_and_rs():
    rng = np.random.default_rng(0)
    q = QAM(64)
    assert abs(np.mean(np.abs(q.points) ** 2) - 1) < 1e-9
    b = rng.integers(0, 2, 600)
    assert (q.demod_hard(q.modulate(b)) == b).all()
    assert ((q.demod_llr(q.modulate(b), 1e-3) > 0) == b).all()
    rs = ReedSolomon(42, 30)
    raw = bytes(range(70))
    cw = bytearray(rs.encode(raw))
    cw[0] ^= 0xFF
    cw[45] ^= 0x0F  # one error in each of the first two blocks
    assert rs.decode(bytes(cw), len(raw)) == raw


def test_traditional_high_snr_error_free():
    for code in (ReedSolomon(42, 30), TurboCode(5)):
        s = TraditionalSystem(HuffmanCoder(SENTS), code, 64, seed=0)
        out, _ = s.transmit(SENTS, 30.0, "awgn")
        assert out == SENTS


def test_constellation_mapper_outputs_exact_points_and_has_gradient():
    from deepsc.constellation import ConstellationMapper
    for name in ("qam4", "qam16", "qam64", "psk8"):
        m = ConstellationMapper(name)
        assert abs((m.points.abs() ** 2).mean().item() - 1) < 1e-5
        z = torch.complex(torch.randn(4, 50), torch.randn(4, 50)).requires_grad_()
        for train in (True, False):
            m.train(train)
            x = m(z)
            gap = (x.detach()[..., None] - m.points).abs().min(-1).values
            assert gap.max() < 1e-5  # always on the constellation
        m.train()
        m(z).real.sum().backward()
        assert z.grad.abs().sum() > 0


def test_digital_deepsc_and_analog_checkpoint_compatible():
    v = Vocab.build(SENTS)
    ds = SentenceDataset(SENTS, v)
    src = ds.collate([ds[i] for i in range(3)])
    analog = DeepSC(len(v), v.pad)
    digital = DeepSC(len(v), v.pad, constellation="qam16", rx_hard=True)
    digital.load_state_dict(analog.state_dict())  # analog weights initialise the digital model
    digital.eval()
    x, mask = digital.transmit(src)
    assert ((x[mask][:, None] - digital.mapper.points).abs().min(-1).values < 1e-5).all()
    out = digital.greedy_decode(src, v.start, v.end, "awgn", 10.0)
    assert out.size(0) == 3
