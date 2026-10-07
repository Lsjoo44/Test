# OCELOT cell detection + tissue segmentation (CAS2250 팀 프로젝트, 개인 파이프라인)

Tissue 예측을 cell branch 에 넘기는 **multi-task DeepLabV3+** 의 최소 파이프라인입니다.
담당 개선 실험 두 가지를 플래그 하나로 바꿀 수 있게 만들었습니다.

- **Stain augmentation**: `--stain none | paired | independent | macenko`
- **Tissue random dropout**: `--tissue_dropout 0.3 --tissue_dropout_mode zero|uniform`

팀 공통 코드가 생기면 `src/stain.py`, `src/augment.py`, `src/model.py` 의 `tissue_prior` / `apply_tissue_dropout`
부분만 떼어서 옮기면 됩니다 (각 파일은 서로 의존이 적게 짜여 있음).

---

## 1. 처음 한 번 (VESSL workspace, `/root` 아래에서)

```bash
cd /root
git clone <이 저장소> ocelot && cd ocelot
bash scripts/setup.sh            # pip 설치 + 공식 평가 코드(ocelot23algo) clone + GPU 확인
```

데이터: https://zenodo.org/records/8417503 에서 약관 동의 후 받아서 `data/` 에 압축 해제.

```bash
unzip ocelot2023_v1.0.1.zip -d data/
python -m src.dataset --data_root data/ocelot2023_v1.0.1    # 구조/라벨 값/개수 점검
```

점검 출력에서 확인할 것: tissue PNG 값이 `[1, 2, 255]`, cell box x0/y0 가 0~768 범위, 장수(train 400 / val 137 / test 126 또는 130).
폴더 이름이 다르면 `configs/*.yaml` 의 `data_root` 를 고치세요.

## 2. 정렬 검증 (학습 전에 꼭!)

```bash
python scripts/visualize_aug.py --data_root data/ocelot2023_v1.0.1 --n 6 --stain paired
python scripts/visualize_aug.py --data_root data/ocelot2023_v1.0.1 --n 4 --stain_grid
```

`results/vis/aug_train_paired.png` 의 한 줄 = 증강 1회:
1. cell crop + GT 점 (TC 파랑, BC 노랑)
2. tissue crop + CA 마스크(초록) + cell 영역 box(빨강)
3. **모델이 실제로 쓰는 `roi_align`** 으로 box 영역을 잘라 키운 tissue 이미지 + 같은 점
4. 같은 방법으로 자른 tissue 마스크 + 같은 점

→ 3번이 1번과 **같은 장면(흐릿한 버전)** 이고, 4번에서 파란 점(TC) 대부분이 초록(CA) 위에 있으면 OK.
`stain_grid.png` 로 염색 증강 세기(`--stain_alpha`, `--stain_beta`)가 너무 세지 않은지 눈으로 확인하세요.

## 3. 학습

```bash
# 짧게 돌려서 시간 측정 (2 epoch)
python -m src.train --config configs/smoke.yaml

# 본 학습 (브라우저 닫아도 계속 돌도록)
nohup python -m src.train --config configs/E0.yaml > logs_E0.txt 2>&1 &
tail -f logs_E0.txt
```

- 매 epoch `checkpoints/<exp>/last.pth` 저장. 중단되면 **같은 명령을 다시 실행**하면 이어서 학습 (`--resume auto` 가 기본).
- `checkpoints/<exp>/best.pth` = val mF1 최고. `log.csv` 에 epoch 별 loss/val 지표/시간.
- 값 일부만 바꾸기: `python -m src.train --config configs/E0.yaml --epochs 60 --batch_size 4`
- OOM 이면 `--batch_size 4` 또는 `--cell_crop 384 --tissue_crop 384`.
- 1 epoch 시간을 보고 `epochs` 를 2~3시간에 맞추세요 (`--repeats 2` 는 epoch 당 각 샘플을 2번 다른 crop 으로 봄).
- Lunit SSL encoder: `--encoder_ckpt lunit_bt` (또는 `lunit_mocov2`, `lunit_swav`). 자동 다운로드하고 입력 정규화도 Lunit 값으로 바뀜.
  encoder 만 낮은 lr: `--encoder_lr_scale 0.1` (1e-4 → 1e-5).

## 4. 추론 + 평가

```bash
# val: 예측 + 후처리(min_distance, TC/BC threshold) 튜닝 → results/E0/postproc.json
python -m src.infer --ckpt checkpoints/E0/best.pth --split val --tune
python -m src.analyze --exp E0 --split val

# test: 맨 마지막에 1회만 (val 에서 튜닝한 후처리 자동 사용)
python -m src.infer --ckpt checkpoints/E0/best.pth --split test
python -m src.analyze --exp E0 --split test
```

`--tta` 를 붙이면 8방향(flip × rot90) 평균.
`analyze` 출력: mF1, TC/BC F1, tissue mIoU, **장기별 mF1**, **세포-조직 조합별 P/R/F1** (TC-on-CA, TC-on-BG, BC-on-CA, BC-on-BG).
결과는 `results/<exp>_<split>.json` 과 `results/summary.csv` (실험마다 한 줄)에 쌓입니다.

**mF1 은 직접 구현하지 않고** `third_party/ocelot23algo/evaluation/eval.py` 의 공식 함수를 불러서 계산합니다
(`src/official_eval.py`). 예측 json 도 공식 형식(`image_{idx}`, 정렬 순서 = `convert_gt_csvs_to_json.py` 와 동일)으로 저장됩니다.

## 5. 실험 목록

| ID | config | 내용 |
|---|---|---|
| E0 | `configs/E0.yaml` | Pred-to-input fusion + 기하 증강 + 약한 photometric (기준점) |
| S1 | `configs/S1.yaml` | E0 + stain **paired** |
| S2 | `configs/S2.yaml` | E0 + stain **independent** |
| S3 | `configs/S3.yaml` | E0 + **Macenko** 정규화 |
| T1 | `configs/T1.yaml` | best-of-S + tissue dropout 0.3 (**S 결과 보고 `stain` 값 수정**) |
| T2 | `configs/T2.yaml` | best-of-S + tissue dropout 0.5 |

시간이 부족하면 E0 → S1 → T1 순서로.

## 6. 구현 메모 (어떻게 동작하나)

- **좌표**: cell 패치(1024px, 0.2MPP)는 tissue 패치 안의 256px 영역. 위치 = metadata 의 `patch_x_offset/patch_y_offset`(cell 패치 중심의 상대좌표) × 1024.
- **증강 순서** (`src/augment.py`): ① cell 이미지·점, tissue 이미지·마스크, **cell box** 에 같은 flip/rot90 → ② cell 512 crop + 그 영역을 포함하는 tissue 512 crop (box 좌표도 따라 이동) → ③ stain → ④ 약한 color jitter(쌍에 같은 값) + blur/noise(각각).
- **Fusion** (`src/model.py`): tissue softmax 확률에서 box 영역을 `roi_align` 으로 잘라 cell 크기로 키운 2채널을 cell RGB 에 concat (입력 5채널). `--detach_tissue` (기본 on) 면 cell loss 가 tissue branch 로 흐르지 않음.
- **Tissue dropout**: 학습 때만, 샘플마다 확률 p 로 그 2채널을 0(zero) 또는 0.5(uniform)로 교체. 추론 때는 항상 사용.
- **Loss**: cell = CE + Dice(BC, TC) on 반지름 7px disk 타깃 (`--cell_target gaussian` 이면 σ=6 heatmap + focal). tissue = CE + Dice, UNK 무시. `L = L_cell + λ·L_tissue`.
- **후처리**: 1 − P(배경) 맵에서 `peak_local_max` → 그 위치의 argmax 클래스, 클래스 확률 = confidence.
- **`--exclude_noisy`**: train 의 152 제외, 242/248/313 tissue 라벨을 전부 UNK 처리. val/test 는 절대 안 건드림.
- **아직 없는 것**: Pred-to-inter-2 fusion (ASPP 뒤 concat), RandStainNA, seed 반복 스크립트.

## 7. 코드 동작 확인 (실제 데이터 없이)

```bash
python scripts/make_dummy_data.py --out data/dummy     # OCELOT 형식의 가짜 데이터
python tests/test_pipeline.py                          # 정렬/증강/stain 테스트
python -m src.train --config configs/smoke.yaml --data_root data/dummy --encoder resnet18 \
    --encoder_weights none --cell_crop 256 --tissue_crop 256 --batch_size 2 --num_workers 2
```

## 폴더 구조

```
src/
  dataset.py        로딩, cell box 계산, disk/gaussian 타깃, 데이터 점검 (__main__)
  augment.py        paired 기하 증강 (+ box 좌표 변환), crop, 약한 photometric
  stain.py          HED jitter (paired/independent), Macenko
  model.py          tissue/cell DeepLabV3+, fusion, tissue dropout, Lunit 가중치 로더, loss
  train.py          플래그/yaml, AMP, ckpt/resume, 매 val_every epoch 공식 mF1
  infer.py          확률맵 → 점, 후처리 튜닝, TTA, 공식 형식 json
  analyze.py        전체/장기별/조합별 지표, summary.csv
  official_eval.py  공식 eval.py 함수 불러오기
scripts/            setup.sh, visualize_aug.py, make_dummy_data.py
configs/            E0, S1, S2, S3, T1, T2, smoke
tests/              test_pipeline.py
```
