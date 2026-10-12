# Project Hub - DeepSC

DeepSC 구현 및 성능 개선을 위한 프로젝트 허브입니다.

| 프로젝트 | 내용 | 주요 결과 |
|---|---|---|
| [**DeepSC-PyTorch**](https://github.com/EHeum/DeepSC-PyTorch) | Xie et al., *"Deep Learning Enabled Semantic Communication Systems,"* IEEE TSP 2021의 시스템 모델 분석과 DeepSC PyTorch 재구현·검증. Huffman/5-bit × RS/Turbo 베이스라인, MI 손실, 전이학습 포함 | AWGN 0 dB BLEU-1 0.61(논문 ≈0.62). 기존 분리 부호화의 cliff effect 대비 완만한 열화 재현 |
| [**DeepSC-Digital**](https://github.com/EHeum/DeepSC-Digital) | DeepJSCC-Q 방식으로 DeepSC 송신 심볼을 M-QAM(4/16/64) 성상점에 제약해 디지털화. soft/hard 수신기, SNR mismatch, cliff 분석 | 64-QAM ≈ analog(12 dB 이상 손실 ≤ 0.003). 디지털화 후에도 cliff 현상 없음 |

## 저장소 구성

| 저장소 | 역할 |
|---|---|
| `DeepSC` (이 저장소) | 프로젝트 안내 |
| `DeepSC-PyTorch` | 논문 재현: analog DeepSC, 기존 분리 부호화 베이스라인, 논문 그림 재현 실험 |
| `DeepSC-Digital` | DeepSC-PyTorch 전체 + constellation 제약 디지털화, cliff 분석 |

각 저장소에 코드, 실험 결과, 재현 스크립트, CI가 들어 있습니다. 하위 두 저장소는 비공개이므로 초대받은 사람만 볼 수 있습니다.
이 저장소에 있던 분리 이전의 작업 이력은 git 기록(`925d1cf` … `1551f8a`)에 남아 있습니다.
