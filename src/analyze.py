"""결과 분석: 전체 / 장기별 / 세포-조직 조합별 지표.

    python -m src.analyze --exp E0 --split val

입력 : results/<exp>/<split>_pred.json, results/<exp>/<split>_tissue/*.png (src.infer 가 만든 것)
출력 : results/<exp>_<split>.json, results/summary.csv (실험별 한 줄씩 누적/갱신)

세포-조직 조합별 F1 (TC-on-CA, TC-on-BG, BC-on-CA, BC-on-BG)
  각 점(GT 와 예측 모두)이 놓인 위치의 'GT tissue' 클래스(CA/BG/UNK)로 나눈 뒤,
  그룹마다 공식 매칭(_calc_scores)으로 Precision/Recall/F1 을 계산한다.
  (경계 근처에서 GT 와 예측이 서로 다른 그룹에 떨어지면 매칭이 안 되는 근사가 있음)
  BG 안의 TC, CA 안의 BC 가 '예외 조합' -> tissue dropout 효과를 보는 핵심 지표.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

from . import official_eval
from .dataset import BC, IGNORE, TC, TISSUE_BG, TISSUE_CA, build_records, read_cell_csv, read_tissue_mask
from .infer import miou_from_confusion, tissue_confusion

REGION_NAMES = {TISSUE_BG: "BG", TISSUE_CA: "CA", IGNORE: "UNK"}


def tissue_class_at(points, box, mask, cell_size=1024):
    """cell 좌표의 점들이 놓인 GT tissue 클래스 (0=BG, 1=CA, 255=UNK)."""
    if len(points) == 0:
        return np.zeros((0,), np.int64)
    x0, y0, x1, y1 = box
    s = (x1 - x0) / cell_size
    pts = np.asarray(points, np.float64)
    tx = np.clip(np.floor(x0 + (pts[:, 0] + 0.5) * s), 0, mask.shape[1] - 1).astype(int)
    ty = np.clip(np.floor(y0 + (pts[:, 1] + 0.5) * s), 0, mask.shape[0] - 1).astype(int)
    return mask[ty, tx].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--data_root", default=None, help="생략하면 infer 때 경로")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--pred", default=None, help="예측 json (기본 results/<exp>/<split>_pred.json)")
    ap.add_argument("--official_eval", default=None)
    args = ap.parse_args()
    official_eval.load(args.official_eval)

    exp_dir = os.path.join(args.results_dir, args.exp)
    info_path = os.path.join(exp_dir, f"{args.split}_infer_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    data_root = args.data_root or info.get("data_root")
    if not data_root:
        raise SystemExit("--data_root 를 지정하세요.")
    records = build_records(data_root, args.split, exclude_noisy=False)
    if info.get("ids"):
        keep = set(info["ids"])
        records = [r for r in records if r.sid in keep]

    preds = official_eval.from_json(args.pred or os.path.join(exp_dir, f"{args.split}_pred.json"))
    preds += [[] for _ in range(len(records) - len(preds))]   # 마지막 이미지들에 예측이 없을 때
    assert len(preds) == len(records), f"예측 이미지 수 {len(preds)} != 데이터 {len(records)}"

    gts, masks = [], []
    for r in records:
        gts.append([tuple(int(v) for v in p) for p in read_cell_csv(r.cell_csv)])
        masks.append(read_tissue_mask(r.tissue_png) if r.tissue_png else None)

    result = {"exp": args.exp, "split": args.split, "n_images": len(records),
              "ckpt": info.get("ckpt"), "epoch": info.get("epoch"), "tta": info.get("tta")}

    # 1) 전체 (공식)
    result["overall"] = official_eval.score(preds, gts)

    # 2) tissue mIoU
    tissue_dir = os.path.join(exp_dir, f"{args.split}_tissue")
    cm = np.zeros((2, 2), np.int64)
    n_t = 0
    for r, m in zip(records, masks):
        p = os.path.join(tissue_dir, f"{r.sid}.png")
        if m is None or not os.path.exists(p):
            continue
        pred = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.int64) - 1   # 1/2 -> 0/1
        cm += tissue_confusion(pred, m)
        n_t += 1
    if n_t:
        result["tissue"] = miou_from_confusion(cm)

    # 3) 장기별
    organs = sorted({r.organ for r in records})
    result["per_organ"] = {}
    for org in organs:
        idx = [i for i, r in enumerate(records) if r.organ == org]
        s = official_eval.score([preds[i] for i in idx], [gts[i] for i in idx])
        s["n_images"] = len(idx)
        result["per_organ"][org] = s

    # 4) 세포-조직 조합별
    combo = {}
    for region in (TISSUE_CA, TISSUE_BG):
        sub_p, sub_g = [], []
        for r, m, pr, gt in zip(records, masks, preds, gts):
            if m is None:
                sub_p.append([])
                sub_g.append([])
                continue
            rp = tissue_class_at([q[:2] for q in pr], r.box, m)
            rg = tissue_class_at([q[:2] for q in gt], r.box, m)
            sub_p.append([q for q, t in zip(pr, rp) if t == region])
            sub_g.append([q for q, t in zip(gt, rg) if t == region])
        s = official_eval.score(sub_p, sub_g)
        rn = REGION_NAMES[region]
        for cls, cn in ((TC, "TC"), (BC, "BC")):
            combo[f"{cn}-on-{rn}"] = {
                "precision": s[f"Pre/{cn}"], "recall": s[f"Rec/{cn}"], "f1": s[f"F1/{cn}"],
                "n_gt": int(sum(sum(1 for q in g if q[2] == cls) for g in sub_g)),
                "n_pred": int(sum(sum(1 for q in p if q[2] == cls) for p in sub_p)),
            }
    result["cell_tissue_combo"] = combo

    out_json = os.path.join(args.results_dir, f"{args.exp}_{args.split}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 5) 통합 CSV (exp, split 기준 upsert)
    o = result["overall"]
    row = {"exp": args.exp, "split": args.split, "epoch": result["epoch"], "tta": result["tta"],
           "mF1": o["mF1"], "F1_TC": o["F1/TC"], "F1_BC": o["F1/BC"],
           "mIoU": result.get("tissue", {}).get("mIoU")}
    for k, v in combo.items():
        row[f"F1_{k}"] = v["f1"]
    for org, s in result["per_organ"].items():
        row[f"mF1_{org}"] = s["mF1"]
    summary = os.path.join(args.results_dir, "summary.csv")
    rows = []
    if os.path.exists(summary):
        with open(summary) as f:
            rows = [r for r in csv.DictReader(f) if not (r["exp"] == args.exp and r["split"] == args.split)]
    rows.append({k: ("" if v is None else v) for k, v in row.items()})
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # 콘솔 출력
    print(f"\n=== {args.exp} / {args.split} ({len(records)}장) ===")
    print(f"mF1 {o['mF1']:.4f} | TC F1 {o['F1/TC']:.4f} | BC F1 {o['F1/BC']:.4f}"
          + (f" | tissue mIoU {result['tissue']['mIoU']:.4f}" if "tissue" in result else ""))
    print("\n[세포-조직 조합]  (n_gt = GT 세포 수)")
    for k, v in combo.items():
        print(f"  {k:10s} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f}  n_gt {v['n_gt']}")
    print("\n[장기별 mF1]")
    for org, s in result["per_organ"].items():
        print(f"  {org:15s} {s['mF1']:.4f}  ({s['n_images']}장)")
    print(f"\n저장: {out_json}, {summary}")


if __name__ == "__main__":
    main()
