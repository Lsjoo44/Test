"""OCELOT 과 같은 폴더 구조/형식의 가짜 데이터 생성 (코드 동작 확인용, 실제 학습용 아님).

cell 패치는 tissue 패치의 해당 256px 영역을 4배 확대해서 만들기 때문에,
증강/정렬이 맞으면 visualize_aug.py 의 'tissue crop' 패널이 cell 패치와 같은 모양이어야 한다.

    python scripts/make_dummy_data.py --out data/dummy
"""
import argparse
import json
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

ORGANS = ["kidney", "head-and-neck", "prostate", "stomach", "endometrium", "bladder"]


def make_pair(rng):
    # tissue: 부드러운 랜덤 blob 을 CA 로
    field = gaussian_filter(rng.standard_normal((1024, 1024)), 40)
    ca = field > np.percentile(field, 55)
    tissue = np.empty((1024, 1024, 3), np.float32)
    tissue[:] = (235, 185, 215)
    tissue[ca] = (175, 110, 195)
    tissue += rng.normal(0, 6, tissue.shape)
    mask = np.where(ca, 2, 1).astype(np.uint8)
    ux, uy = rng.integers(0, 900, 2)
    mask[uy:uy + 60, ux:ux + 90] = 255              # UNK 영역

    # cell 패치 중심 (tissue 좌표, cell 영역이 tissue 안에 완전히 들어가도록)
    cx, cy = rng.integers(128, 896, 2)
    x0, y0 = cx - 128, cy - 128
    cell = cv2.resize(tissue[y0:y0 + 256, x0:x0 + 256], (1024, 1024), interpolation=cv2.INTER_CUBIC)

    pts = []
    for _ in range(rng.integers(60, 160)):
        x, y = rng.integers(8, 1016, 2)
        on_ca = ca[int(y0 + (y + 0.5) / 4), int(x0 + (x + 0.5) / 4)]
        cls = 2 if (on_ca ^ (rng.random() < 0.1)) else 1   # 10% 예외 조합
        r, col = (8, (70, 20, 110)) if cls == 2 else (5, (30, 30, 90))
        cv2.circle(cell, (int(x), int(y)), r, col, -1)
        cv2.circle(tissue, (int(x0 + x / 4), int(y0 + y / 4)), 1 if cls == 1 else 2, col, -1)
        pts.append((int(x), int(y), cls))
    meta = {"patch_x_offset": float(cx / 1024), "patch_y_offset": float(cy / 1024)}
    clip = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    return clip(cell), clip(tissue), mask, pts, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy")
    ap.add_argument("--n", type=int, nargs=3, default=[8, 4, 4], help="train val test 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    meta_all = {}
    sid = 1
    for split, n in zip(("train", "val", "test"), args.n):
        for d in ("images/{}/cell", "images/{}/tissue", "annotations/{}/cell", "annotations/{}/tissue"):
            os.makedirs(os.path.join(args.out, d.format(split)), exist_ok=True)
        for _ in range(n):
            cell, tissue, mask, pts, meta = make_pair(rng)
            name = f"{sid:03d}"
            cv2.imwrite(f"{args.out}/images/{split}/cell/{name}.jpg", cv2.cvtColor(cell, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{args.out}/images/{split}/tissue/{name}.jpg", cv2.cvtColor(tissue, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{args.out}/annotations/{split}/tissue/{name}.png", mask)
            with open(f"{args.out}/annotations/{split}/cell/{name}.csv", "w") as f:
                f.write("".join(f"{x},{y},{c}\n" for x, y, c in pts))
            meta_all[name] = {"slide_name": f"dummy_{sid}", "organ": ORGANS[sid % len(ORGANS)],
                              "subset": split, "mpp_x": 0.25, "mpp_y": 0.25,
                              "cell": {"resized_mpp_x": 0.2, "resized_mpp_y": 0.2},
                              "tissue": {"resized_mpp_x": 0.8, "resized_mpp_y": 0.8}, **meta}
            sid += 1
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump({"dataset_name": "dummy", "sample_pairs": meta_all}, f, indent=2)
    print(f"dummy data -> {args.out} ({sum(args.n)} pairs)")


if __name__ == "__main__":
    main()
