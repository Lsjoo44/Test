> **구현 상태 (이 저장소)** — 9장 작업 순서 1~4 의 코드는 작성 완료, 가짜 데이터로 end-to-end 검증까지 함.
> 실제 OCELOT 데이터로는 아직 실행 안 됨 → 다음 할 일: `python -m src.dataset --data_root ...` 로 실제 구조 확인 → `scripts/visualize_aug.py` 정렬 확인 → `configs/smoke.yaml` 로 시간 측정 → E0.
> 사용법은 README.md 참고. 팀 공통 기준(데이터 버전, 학습 조건)이 정해지면 configs/*.yaml 값을 바꾼다.

# OCELOT 팀 프로젝트 — 코드 작업 인수인계 (Handoff)

> 이 문서는 Claude Code 세션에 그대로 붙여 넣거나, 저장소 루트에 `CLAUDE.md`로 두고 쓰기 위한 정리본입니다.
> 사용자: 박주석 (연세대 CAS2250 인공지능개론, AI 입문자 · 시간이 촉박함). **설명은 쉽게, 한국어로, 수학/ML 용어는 영어 표기 유지.**

---

## 0. 지금 당장 하려는 일 (TL;DR)

**팀 공통 코드가 아직 없으므로, 최소한의 독립 파이프라인을 먼저 만들어 학습을 돌려본다.**
담당 개선 방향 두 가지를 실험한다.

1. **Stain augmentation** (H&E 염색 색감 증강) — paired vs independent 비교 포함
2. **Tissue random dropout** — tissue 예측을 cell branch에 넘기는 구조에서, 학습 중 tissue 정보를 확률적으로 지움

나중에 팀 GitHub 공통 코드가 올라오면 **증강/드롭아웃 모듈만 떼어서** 내 브랜치로 옮길 수 있게 모듈화한다.

---

## 1. 프로젝트 개요

- 과목: CAS2250 Introduction to AI, Medical AI 팀 프로젝트 (5인)
- 과제: OCELOT 데이터로 **cell detection (TC/BC 점)** + **tissue segmentation (CA/BG 마스크)** 모델 학습·평가
- 강의 권장 Base: Lunit SSL 사전학습 encoder(ResNet-50 권장, ViT-S DINO 옵션) + 직접 만든 head, multi-task
- 평가
  - Cell: **mF1** — 예측점과 GT가 **~3µm(≈15px)** 이내 + 클래스 일치 시 TP, confidence 순 greedy 1:1 매칭, TC/BC F1 평균
  - Tissue: **mIoU**, UNK 픽셀은 분자·분모 모두 제외
- 공식 평가 코드: https://github.com/lunit-io/ocelot23algo/tree/main/evaluation (직접 구현하지 말고 이걸 사용)

### 참고 수치 (OCELOT test, cell mF1)
| 모델 | test mF1 |
|---|---|
| Cell-only (DeepLabV3+ R34, ImageNet) | 64.44 |
| Pred-to-input (tissue 예측을 cell 입력에 concat) | 69.65 |
| **Pred-to-inter-2 (ASPP 뒤 feature에 concat, 논문 최고)** | **71.23** |
| 챌린지 1위 (SAM+CellViT 앙상블) | 72.44 |
| Tissue-label leaking (GT tissue 입력, 상한 참고용) | 74.20 |

Tissue mIoU 참고: 논문 81.97 (강의자료 표).

---

## 2. 데이터 (OCELOT)

- 다운로드: https://zenodo.org/records/8417503 (약 303MB, 약관 동의 필요)
- **버전 주의**: v1.0.1은 under-annotation으로 test 4건(586, 589, 609, 615) 제외 → test 126장. 강의 안내는 130장. **팀과 버전 통일 필요.**
- Split: train 400 / val 137 / test 130 (WSI 단위 분할, 6개 장기: kidney, head-and-neck, prostate, stomach, endometrium, bladder)
- Cell 패치: 1024×1024, 0.2 MPP
- Tissue 패치: 4096×4096 원본을 1/4 다운샘플 → 1024×1024 (≈0.8 MPP)
  - ⇒ **cell 패치는 tissue 패치 안의 256×256 영역**에 해당. 위치는 metadata의 "tissue 패치 내 cell 패치 중심 상대좌표"로 구함
- 라벨 형식 (공식 문서 기준, **실제 다운로드 후 반드시 확인**)
  - Cell: CSV, `x, y, class` — class **1 = BC, 2 = TC**, 좌표 (0,0)~(1023,1023)
  - Tissue: PNG 마스크 — **1 = BG, 2 = CA, 255 = UNK**
  - metadata.json: 장기, MPP, cell 패치의 상대 위치 등
- 통계 (train): TC 43.8K, BC 23.6K (BC가 적음 → BC F1이 낮게 나옴). TC의 ~93%가 CA 안에 위치.

### 라벨 불일치 의심 샘플 (팀원 민보성 발견, train)
| ID | 장기 | 내용 | 처리 제안 |
|---|---|---|---|
| 248 | bladder | tissue 전체 BG, cell TC 365 / BC 0 | tissue 라벨만 UNK 처리 |
| 242 | bladder | tissue 전체 BG, cell TC 55 / BC 0 | tissue 라벨만 UNK 처리 |
| 313 | prostate | tissue 전체 BG, cell TC 183 / BC 153 | tissue 라벨만 UNK 처리 |
| 152 | endometrium | tissue 100% UNK, cell BC 10개만 (under-annotation) | 학습 제외 또는 cell loss 제외 |

→ 코드에 `--exclude_noisy` 같은 플래그로 on/off 가능하게. **val/test는 절대 수정 금지.**

---

## 3. 모델 / 학습 사양 (최소 파이프라인)

### 구조
- 라이브러리: `segmentation_models_pytorch` (DeepLabV3+)
- Encoder: 우선 `resnet50` (ImageNet) 로 시작 → 이후 **Lunit SSL ResNet-50 가중치**로 교체 가능하게
  - 가중치: https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights (encoder만 제공, key 이름 매핑 확인 필요)
- Tissue branch: tissue 패치(1024, 학습 시 crop 가능) → 3클래스(BG/CA, UNK는 ignore) 로짓
- Cell branch: cell 패치 → 3채널(배경/BC/TC) 분할맵
- **Fusion (tissue → cell)**: tissue 예측 확률(softmax)에서 cell 영역(256×256)을 crop → 1024로 4× upsample → cell branch에 concat
  - 1차: **Pred-to-input** (cell 입력 RGB에 tissue 확률 2채널 concat → 입력 5채널) — 구현이 가장 단순
  - 2차(여유 시): Pred-to-inter-2 (ASPP 출력 해상도로 resize 후 concat)
- 학습 방식: joint (L_total = L_cell + λ·L_tissue, λ=1.0), tissue 예측은 fusion에 넣을 때 `detach()` 여부를 옵션으로

### Cell 라벨 → 학습 타깃
- 기본: 각 점에 반지름 7px 원판(disk), 클래스값 채움 (논문 방식) + Dice(+CE)
- 옵션: Gaussian heatmap (σ≈1.14µm≈6px) + MSE/Focal (챌린지 2·3위 방식)

### 추론 → 점
- 클래스 확률맵 → `skimage.feature.peak_local_max` (min_distance, threshold는 val에서 튜닝) → argmax 클래스, 확률을 confidence로 → 공식 평가 코드 입력 형식으로 저장

### 학습 설정 (기본값)
- AdamW lr 1e-4, cosine annealing (강의 권장). Encoder는 낮은 lr(예: 1e-5) 옵션
- 논문 관찰: cell branch가 tissue보다 먼저 overfit → branch별 lr/dropout 분리 옵션
- Batch: 3090 24GB 기준 crop 512 / batch 8 정도부터 시작 (OOM 시 조정)
- AMP(mixed precision) 사용
- 1회 학습 목표 2~3시간 (팀원 측정: RN50 low-lr 3h, LoRA 2.5h, frozen 2h)
- **Checkpoint 매 epoch 저장 + resume 지원** (VESSL 24시간 자동 종료 대비)
- 모델 선택은 **val 기준만**, test는 맨 마지막 1회

---

## 4. 구현할 개선 모듈 (내 담당)

### 4-1. 공통 기하 증강 (모든 실험의 기본)
- flip H/V, rot90 — **cell 이미지·cell 점·tissue 이미지·tissue 마스크에 같은 변환**
- ⚠️ **함정**: tissue 패치를 뒤집거나 돌리면 **tissue 안 cell 영역의 위치(상대좌표)도 같이 변환**해야 fusion이 맞는 영역을 가리킴
- 시각화 검증 스크립트 필수: 증강 후 이미지 위에 점/마스크/cell 영역 박스 overlay → PNG 저장해 눈으로 확인

### 4-2. Stain augmentation (`stain.py`)
- **HED jitter**: `skimage.color.rgb2hed` → 각 채널에 `α∈[1-σ,1+σ]` 곱, `β∈[-σ,σ]` 더함 (σ≈0.03~0.05로 약하게) → `hed2rgb`
- (옵션) RandStainNA, Macenko normalization(`torchstain`)
- 모드 플래그
  - `--stain none`
  - `--stain paired` : cell/tissue 쌍에 **같은 α,β** (같은 슬라이드라 현실적)
  - `--stain independent` : cell/tissue에 **각각 다른 α,β**
  - `--stain macenko` : 정규화(색 통일) — 증강과 비교용
- 약한 color jitter / blur / noise는 기본 photometric으로 포함 (강하면 비현실적 → 약하게)

### 4-3. Tissue random dropout (`fusion` 내부)
- 학습 시 확률 `p`로 cell branch에 들어가는 tissue 확률맵을 **0(또는 균등분포 0.5)** 으로 대체
- 플래그: `--tissue_dropout p` (실험값 0.0 / 0.3 / 0.5), 모드 `zero|uniform`
- 추론 시에는 항상 tissue 정보 사용
- 목적: tissue prior 과의존 완화 → **예외 조합(BG 안 TC, CA 안 BC)** 성능 개선 기대
  - 근거: OCELOT 2023 챌린지 분석 — 상위팀들이 예외 조합에서 크게 하락 (arXiv 2509.09153)

### 4-4. 분석 스크립트 (`analyze.py`)
- 출력: val mF1, **TC F1, BC F1**, tissue mIoU, **장기별 mF1**
- **세포–조직 조합별 F1**: 각 GT 세포가 놓인 tissue GT 클래스(CA/BG)로 나눠 TC-on-CA, TC-on-BG, BC-on-CA, BC-on-BG 별 recall/F1
- 결과를 `results/<exp_name>.json` + 통합 CSV 표로

---

## 5. 실험 계획 (우선순위 순)

| ID | 설정 | 비교 목적 |
|---|---|---|
| E0 | fusion(Pred-to-input) + 기하 증강 + 약한 photometric | 내 기준점 |
| S1 | E0 + stain **paired** | 염색 증강 효과 |
| S2 | E0 + stain **independent** | paired vs independent |
| S3 | E0 + **Macenko** 정규화 | 통일 vs 다양화 |
| T1 | (best of S) + tissue dropout p=0.3 | 과의존 완화 |
| T2 | (best of S) + tissue dropout p=0.5 | p 민감도 |

- 시간 부족 시: **E0 → S1 → T1** 3개만 먼저.
- 여유 시: 최종 설정 seed 3개 반복, TTA(8방향 평균), TC/BC별 threshold 튜닝.

---

## 6. 실행 환경 (학교 VESSL)

- GPU: **RTX 3090 24GB** (Resource `gpu-1-128` = GPU 1장 사용), 0.10 크레딧/시간, 최대 24시간
- Workspace: PyTorch Managed Image, **Disk 50GB** (생성 후 변경 불가)
- **`/root`만 영구 저장** → 코드·데이터·체크포인트 전부 `/root/ocelot/` 아래
- Initialization script:
  ```bash
  pip install -q segmentation-models-pytorch albumentations scikit-image opencv-python-headless pandas matplotlib tqdm torchstain
  ```
- 학습은 `tmux` 또는 `nohup ... &`로 실행 (브라우저 닫아도 유지)
- 다 쓰면 Workspace **Stop** (켜두면 크레딧 소모)
- 첫 확인 명령:
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  df -h /root
  ```

### 권장 폴더 구조
```
/root/ocelot/
├── data/                # Zenodo 압축 해제
├── src/
│   ├── dataset.py       # 로딩, cell 위치 계산, 타깃 생성
│   ├── augment.py       # paired 기하 증강 (+ cell 위치 좌표 변환)
│   ├── stain.py         # HED jitter / Macenko
│   ├── model.py         # tissue/cell DeepLabV3+ + fusion + tissue dropout
│   ├── train.py         # argparse 플래그로 실험 전환, ckpt/resume, AMP
│   ├── infer.py         # peak 추출 → 공식 평가 형식
│   └── analyze.py       # 지표/장기별/조합별 F1
├── scripts/visualize_aug.py
├── configs/             # 실험별 yaml (E0, S1, S2, S3, T1, T2)
├── checkpoints/
└── results/
```

---

## 7. 팀 상황 / 협업 규칙

- 역할(팀장 이득모 확정안)
  - 1명: 데이터셋·EDA + 1·2차 보고서 + 최종 발표
  - 1명: Base 실험 + 결과·문제점 분석 + **개선 결과 비교·결합 실험**
  - 3명: 각자 다른 방향 성능 개선 (**나: 병리 특화 증강 + 예외 사례 강건성**)
- 팀장 피드백: 데이터 증강·예외 사례 타깃 증강·tissue random dropout = 좋은 후보. **Tissue dropout은 tissue→cell 전달 구조에서 비교.** Generated data·pseudo-label은 우선순위 낮음. Dense 모델(민보성)은 개선 방향 중 하나로 Lunit base와 같은 기준에서 비교.
- 민보성: dense 모델 작업 중 증강/파이프라인 실험을 많이 해봄 → 공유 예정. **일반 증강은 중복 피하고 그의 결과 활용.**
- 팀 GitHub 저장소 생성 예정 → 공통 코드 기반 **개인 브랜치** 작업, 공통 코드 변경 전 공유
- 데이터 버전, 공식 평가 코드, 기본 학습 조건은 팀 공통 기준을 따른다 (나오면 이 문서의 3장 값을 그걸로 교체)

## 8. 일정
| 날짜 | 내용 |
|---|---|
| 10/8 | 회의 (역할·base·평가 기준 확정) |
| ~10/16 | Base 실험 결과 (Base 담당) / 나: 파이프라인 + E0 |
| 10/19 | 1차 보고서 — 나는 "향후 계획" 3~5줄 전달 |
| ~11/13 | 개선 실험 완료 (S1~T2) + 결과표 |
| 11/16 | 2차 보고서·발표 |

## 9. 작업 순서 (Claude Code에게 요청할 순서)
1. 데이터 압축 해제 후 실제 폴더 구조·파일 형식 확인 → `dataset.py`
2. `augment.py` + `scripts/visualize_aug.py`로 **정렬 검증** (점·마스크·cell 박스가 맞는지)
3. `model.py` (Pred-to-input fusion + tissue dropout), `train.py` (플래그, AMP, ckpt/resume)
4. 공식 평가 코드 연결 `infer.py`, `analyze.py`
5. 짧은 smoke test (2~3 epoch) → 시간 측정 → E0 본 학습
6. S1 → T1 순으로 학습, 결과표 갱신
