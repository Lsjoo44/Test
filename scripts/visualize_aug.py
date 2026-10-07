"""증강 후 정렬 검증 그림 저장.

    python scripts/visualize_aug.py --data_root data/ocelot2023_v1.0.1 --n 6 --stain paired
    python scripts/visualize_aug.py --data_root ... --stain_grid      # 염색 모드 비교 그림

한 줄(샘플 x 증강 1회)에 4개 패널:
  1) cell crop + GT 점 (TC 파랑, BC 노랑)
  2) tissue crop + CA 마스크(초록) + cell 영역 box(빨강)
  3) 모델이 실제로 쓰는 roi_align 으로 tissue 이미지에서 box 영역을 잘라 cell 크기로 키운 것 + 같은 점
  4) 같은 방법으로 자른 tissue GT 마스크 + 같은 점
=> 3) 이 1) 과 같은 장면(흐릿한 버전)이고, 4) 에서 TC 가 대부분 초록(CA) 위에 있으면 정렬 OK.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.augment import TrainAugment          # noqa: E402
from src.dataset import build_records, load_sample  # noqa: E402
from src.stain import MacenkoNormalizer, StainAugment  # noqa: E402

TC_COLOR, BC_COLOR = (0, 120, 255), (255, 220, 0)


def draw_points(img, pts, r=6):
    img = img.copy()
    for x, y, c in pts:
        cv2.circle(img, (int(x), int(y)), r, TC_COLOR if c == 2 else BC_COLOR, 2)
    return img


def overlay_mask(img, mask):
    out = img.astype(np.float32)
    ca = mask == 1
    out[ca] = out[ca] * 0.6 + np.array([0, 200, 0]) * 0.4
    unk = mask == 255
    out[unk] = out[unk] * 0.6 + np.array([128, 128, 128]) * 0.4
    return out.astype(np.uint8)


def roi(img, box, size, nearest=False):
    t = torch.from_numpy(np.ascontiguousarray(img)).float()
    t = t[None, None] if t.ndim == 2 else t.permute(2, 0, 1)[None]
    rois = torch.tensor([[0, *[float(v) for v in box]]])
    if nearest:   # 마스크: 1-샘플 bilinear 로 근사 후 반올림
        o = roi_align(t, rois, (size, size), 1.0, sampling_ratio=1, aligned=True)
        return o[0, 0].round().numpy().astype(np.uint8)
    o = roi_align(t, rois, (size, size), 1.0, sampling_ratio=2, aligned=True)
    return o[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def fit(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=6, help="샘플 수")
    ap.add_argument("--draws", type=int, default=2, help="샘플당 증강 횟수")
    ap.add_argument("--cell_crop", type=int, default=512)
    ap.add_argument("--tissue_crop", type=int, default=512)
    ap.add_argument("--stain", default="none", choices=["none", "paired", "independent", "macenko"])
    ap.add_argument("--stain_alpha", type=float, default=0.05)
    ap.add_argument("--stain_beta", type=float, default=0.01)
    ap.add_argument("--stain_grid", action="store_true")
    ap.add_argument("--out", default="results/vis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    recs = build_records(args.data_root, args.split)
    recs = [recs[i] for i in rng.choice(len(recs), min(args.n, len(recs)), replace=False)]

    if args.stain_grid:
        return stain_grid(recs, args, rng)

    norm = MacenkoNormalizer() if args.stain == "macenko" else None
    aug = TrainAugment(args.cell_crop, args.tissue_crop, True, True, args.stain,
                       args.stain_alpha, args.stain_beta)
    P = 384
    rows = []
    for rec in recs:
        for d in range(args.draws):
            s = aug(load_sample(rec, norm), rng)
            C = s["cell_img"].shape[0]
            p1 = draw_points(s["cell_img"], s["points"])
            t = overlay_mask(s["tissue_img"], s["tissue_mask"])
            x0, y0, x1, y1 = [int(round(v)) for v in s["box"]]
            cv2.rectangle(t, (x0, y0), (x1, y1), (255, 0, 0), 2)
            p3 = draw_points(roi(s["tissue_img"], s["box"], C), s["points"])
            p4 = draw_points(overlay_mask(np.full((C, C, 3), 255, np.uint8),
                                          roi(s["tissue_mask"], s["box"], C, nearest=True)), s["points"])
            rows.append(np.concatenate([
                label(fit(p1, P), f"{rec.sid} cell crop"),
                label(fit(t, P), "tissue crop + box"),
                label(fit(p3, P), "roi_align(tissue)"),
                label(fit(p4, P), "roi_align(mask)")], 1))
    path = os.path.join(args.out, f"aug_{args.split}_{args.stain}.png")
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"저장: {path}")


def stain_grid(recs, args, rng):
    P = 320
    mac = MacenkoNormalizer()
    rows = []
    for rec in recs:
        s = load_sample(rec)
        c, t = s["cell_img"][:512, :512], s["tissue_img"]
        panels = [label(fit(c, P), f"{rec.sid} cell orig"), label(fit(t, P), "tissue orig")]
        for mode in ("paired", "independent"):
            sa = StainAugment(mode, args.stain_alpha, args.stain_beta)
            c2, t2 = sa(c, t, rng)
            panels += [label(fit(c2, P), f"cell {mode}"), label(fit(t2, P), f"tissue {mode}")]
        panels += [label(fit(mac(c), P), "cell macenko"), label(fit(mac(t), P), "tissue macenko")]
        rows.append(np.concatenate(panels, 1))
    path = os.path.join(args.out, "stain_grid.png")
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
