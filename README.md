# DeepSC

[![CI](https://github.com/EHeum/DeepSC/actions/workflows/ci.yml/badge.svg)](https://github.com/EHeum/DeepSC/actions/workflows/ci.yml)

DeepSC(Deep Learning Enabled Semantic Communication) 구현과 성능 개선을 위한 저장소입니다.

> H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, **"Deep Learning Enabled Semantic Communication Systems,"**
> *IEEE Trans. Signal Processing*, vol. 69, pp. 2663–2675, 2021.

## 실험 문서

| 문서 | 내용 | 주요 결과 |
|---|---|---|
| [01 논문 재현](docs/01_paper_reproduction.md) | 시스템 모델 분석, DeepSC PyTorch 재구현, Huffman/5-bit × RS/Turbo 베이스라인, MI 손실, 단어당 심볼 수, 전이학습 (Fig. 6, 8, 9, 10, 13, Table II, III) | AWGN 0 dB BLEU-1 0.61(논문 ≈ 0.62). 기존 분리 부호화의 cliff effect 대비 완만한 열화 재현 |
| [02 디지털화](docs/02_digital_deepsc.md) | DeepJSCC-Q 방식으로 송신 심볼을 4/16/64-QAM 성상점에 제약. soft/hard 수신기, SNR mismatch, MI 포화, cliff 정량 분석 | 64-QAM ≈ analog(12 dB 이상 손실 ≤ 0.003). 디지털화 후에도 cliff 현상 없음 |
| [03 OFDM](docs/03_ofdm.md) | 송신 심볼을 CP-OFDM(64 부반송파, 8탭 다중경로, ZF) 부반송파에 실어 전송. 전이학습, analog·16-QAM·기존 방식 비교, 부반송파 interleaving, i.i.d. Rayleigh로 페이딩 손실과 상관 손실 분해 | 0 dB BLEU-1 0.48, cliff 없음. AWGN 대비 손실은 전부 페이딩 손실이고, 상관은 오히려 도움(interleaving 시 0 dB 0.48 → 0.41) |

| 요약 (AWGN, BLEU-1) | 0 dB | 12 dB | 24 dB | cliff |
|---|---|---|---|---|
| Analog DeepSC | 0.65 | 0.88 | 0.89 | 없음 |
| DeepSC 64-QAM | 0.62 | 0.88 | 0.88 | 없음 |
| DeepSC 16-QAM | 0.62 | 0.87 | 0.87 | 없음 |
| DeepSC 4-QAM | 0.55 | 0.83 | 0.84 | 없음 |
| Huffman + Turbo (64-QAM) | 0.01 | 1.00 | 1.00 | 6→9 dB에서 급변 |

모든 수치는 Europarl 영어 샘플(학습 11,687 / test 1,298문장, 기존 방식은 500문장)을 CPU로 학습·평가한 결과입니다.
데이터 규모와 구현상의 선택은 [01 논문 재현 2.3절](docs/01_paper_reproduction.md#23-구현상의-선택과-논문과의-차이-중요)에 정리돼 있습니다.

## 저장소 구성

```
deepsc/                  핵심 라이브러리
  data.py                Europarl 전처리(4–30 단어), 어휘집, Dataset
  models.py              DeepSC 송수신기, MINE, MI 하한 (constellation / hard 수신기 옵션 포함)
  channels.py            AWGN / Rayleigh(block, i.i.d.) / Rician / Erasure / CP-OFDM 다중경로 채널, 전력 정규화
  constellation.py       M-QAM soft-to-hard 양자화 (DeepJSCC-Q)
  engine.py              학습(Algorithm 1–4)·평가 루프
  metrics.py             BLEU, BERT sentence similarity
  baselines.py           Huffman / 5-bit + RS / Turbo(log-MAP) + Gray QAM
scripts/
  prepare_data.py        데이터 다운로드·전처리
  train.py               학습 (--channel, --mi-lambda, --symbols-per-word, --init/--freeze, --constellation)
  evaluate.py            SNR별 BLEU, 문장 예시, 처리 시간, 송신 심볼 통계 (--rx-hard)
  eval_baselines.py      기존 분리 부호화 평가
  estimate_mi.py         학습된 송신기의 I(X;Y)
  plot_results.py        논문 재현 그림          → results/paper/
  plot_digital.py        디지털화 그림, cliff 지표 → results/digital/
  run_all.sh             논문 재현 실험 전체
  run_digital.sh         디지털화 실험 전체
  run_ofdm.sh            OFDM 실험 전체 (plot_ofdm.py로 그림)
  run_fading_decomp.sh   페이딩 손실 / 상관 손실 분해 실험 (fading_diversity.py: 채널만 본 다이버시티)
tests/test_deepsc.py     단위 테스트
docs/                    실험 문서
results/paper/           논문 재현 결과(JSON, 그림, 학습 곡선)
results/digital/         디지털화 결과
results/ofdm/            OFDM 결과
logs/                    학습·평가 로그
data/europarl_en.json    전처리된 데이터
checkpoints/             학습된 모델(.pt, 저장소에 포함하지 않음)
```

## 실행

```bash
git clone https://github.com/EHeum/DeepSC.git && cd DeepSC
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # GPU라면 PyTorch 공식 안내에 맞는 wheel
pip install -r requirements-dev.txt

# analog DeepSC 학습·평가
python3 scripts/train.py --channel awgn --epochs 40 --out checkpoints/deepsc_awgn.pt
python3 scripts/evaluate.py --ckpt checkpoints/deepsc_awgn.pt --out results/paper/deepsc_awgn.json

# 16-QAM 디지털 DeepSC (analog 모델에서 미세조정)
python3 scripts/train.py --init checkpoints/deepsc_awgn.pt --epochs 15 --lr 5e-4 \
    --constellation qam16 --out checkpoints/dsc_qam16_ft.pt
python3 scripts/evaluate.py --ckpt checkpoints/dsc_qam16_ft.pt --snrs=-6,-3,0,3,6,9,12,15,18,21,24 \
    --out results/digital/qam16_ft.json
```

전체 재현은 `scripts/run_all.sh`(논문 재현, CPU 4코어 약 6시간)와 `scripts/run_digital.sh`(디지털화, 약 4시간)로 합니다.
CUDA GPU가 있으면 자동으로 사용합니다. 전체 Europarl은 `python3 scripts/prepare_data.py --raw <디렉터리>`로 넣을 수 있습니다.

## 개발 환경

```bash
ruff check deepsc scripts tests    # lint
pytest -q                          # 단위 테스트
```

* **CI** (`.github/workflows/ci.yml`): `main` push와 PR마다 GitHub Actions(ubuntu, Python 3.11, CPU PyTorch)에서
  ruff, pytest, 그리고 analog → 16-QAM 1 epoch 학습·평가 스모크 테스트를 돌립니다.
* **Claude Code 클라우드 세션** (`.claude/hooks/session-start.sh`): 원격 세션이 시작될 때 의존성을 자동으로 설치합니다.
* 학습된 체크포인트(`checkpoints/*.pt`, 개당 약 22 MB)는 저장소에 포함하지 않습니다.
