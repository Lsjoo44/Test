"""추론: 확률맵 -> 점 추출 -> 공식 평가 형식 json 저장 (+ val 에서 후처리 파라미터 튜닝).

사용 예
    # val: 예측 + threshold/min_distance 튜닝 (results/<exp>/postproc.json 저장)
    python -m src.infer --ckpt checkpoints/E0/best.pth --split val --tune
    # test: val 에서 튜닝한 값으로 1회만
    python -m src.infer --ckpt checkpoints/E0/best.pth --split test
    # 8방향 TTA
    python -m src.infer --ckpt checkpoints/E0/best.pth --split val --tta

출력 (results/<exp>/)
    <split>_pred.json       공식 eval 입력 형식 cell 예측
    <split>_tissue/<id>.png tissue 예측 마스크 (1=BG, 2=CA, 원본 GT 와 같은 값)
    gt_<split>.json         공식 형식 GT (convert_gt_csvs_to_json.py 와 같은 순서)
    postproc.json           (--tune) 튜닝된 후처리 파라미터
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader

from . import official_eval
from .augment import dihedral_box
from .dataset import BC, TC, OcelotEvalDataset, build_records
from .model import build_model
from .stain import MacenkoNormalizer

DEFAULT_POST = {"min_distance": 5, "thr_fg": 0.1, "thr": {"1": 0.3, "2": 0.3}, "smooth_sigma": 1.0}


# ---------------------------------------------------------------------------
# 모델 실행
# ---------------------------------------------------------------------------
def _tensor_dihedral(t, k, flip):
    if flip:
        t = torch.flip(t, dims=(-1,))
    return torch.rot90(t, k, dims=(-2, -1))


def _tensor_dihedral_inv(t, k, flip):
    t = torch.rot90(t, -k, dims=(-2, -1))
    if flip:
        t = torch.flip(t, dims=(-1,))
    return t


@torch.no_grad()
def predict_pair(model, cell, tissue, box, cell_target="disk", tta=False, amp=True):
    """cell, tissue: (3, H, W) uint8 tensor, box: (4,) numpy.
    반환: cell_prob (C, H, W) numpy float32, tissue_prob (2, H, W) numpy float32.
      disk     : C=3 (bg, BC, TC) softmax
      gaussian : C=2 (BC, TC) sigmoid"""
    dev = next(model.parameters()).device
    c = cell[None].to(dev).float() / 255
    t = tissue[None].to(dev).float() / 255
    T = t.shape[-1]
    views = [(k, f) for f in (False, True) for k in range(4)] if tta else [(0, False)]
    cp_sum, tp_sum = 0, 0
    for k, f in views:
        b = torch.from_numpy(dihedral_box(box, T, k, f))[None].to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            cl, tl = model(_tensor_dihedral(c, k, f), _tensor_dihedral(t, k, f), b)
        cp = torch.softmax(cl.float(), 1) if cell_target == "disk" else torch.sigmoid(cl.float())
        tp = torch.softmax(tl.float(), 1)
        cp_sum = cp_sum + _tensor_dihedral_inv(cp, k, f)
        tp_sum = tp_sum + _tensor_dihedral_inv(tp, k, f)
    n = len(views)
    return (cp_sum / n)[0].cpu().numpy(), (tp_sum / n)[0].cpu().numpy()


# ---------------------------------------------------------------------------
# 후처리: 확률맵 -> 점
# ---------------------------------------------------------------------------
def candidate_peaks(cell_prob, cell_target="disk", min_distance=5, thr_fg=0.1, smooth_sigma=1.0):
    """class 무관 peak 후보. 반환 (N, 4): x, y, cls, conf(=해당 class 확률)."""
    cell_prob = cell_prob.astype(np.float32, copy=False)
    if cell_target == "disk":
        cls_prob = cell_prob[1:3]
        fg = 1.0 - cell_prob[0]
    else:
        cls_prob = cell_prob
        fg = cell_prob.max(0)
    if smooth_sigma > 0:
        fg = gaussian_filter(fg, smooth_sigma)
    yx = peak_local_max(fg, min_distance=int(min_distance), threshold_abs=thr_fg, exclude_border=False)
    if len(yx) == 0:
        return np.zeros((0, 4), np.float32)
    ys, xs = yx[:, 0], yx[:, 1]
    cp = cls_prob[:, ys, xs]                  # (2, N)
    cls = cp.argmax(0) + 1                    # 1=BC, 2=TC
    conf = cp.max(0)
    return np.stack([xs, ys, cls, conf], 1).astype(np.float32)


def filter_points(cands, thr):
    """class 별 threshold 적용 -> 공식 형식 튜플 리스트."""
    out = []
    for x, y, c, p in cands:
        if p >= thr[str(int(c))]:
            out.append((int(x), int(y), int(c), float(min(max(p, 0.0), 1.0))))
    return out


def tune_postproc(cell_probs, gts, cell_target, grid_md=(3, 5, 7, 9),
                  grid_thr=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), smooth_sigma=1.0, thr_fg=0.05):
    """val 에서 min_distance 와 class 별 threshold 를 공식 F1 기준으로 grid search.
    peak 위치는 class 와 무관하므로 class 별 threshold 는 독립적으로 고를 수 있다."""
    best = None
    log = []
    for md in grid_md:
        cands = [candidate_peaks(p, cell_target, md, thr_fg, smooth_sigma) for p in cell_probs]
        thr_best, f1s = {}, {}
        for cls in (BC, TC):
            bf, bt = -1, None
            for th in grid_thr:
                preds = [[(int(x), int(y), int(c), float(p)) for x, y, c, p in cd if c == cls and p >= th]
                         for cd in cands]
                s = official_eval.score(preds, gts, classes=[cls])
                f1 = list(v for k, v in s.items() if k.startswith("F1/"))[0]
                log.append({"min_distance": md, "cls": cls, "thr": th, "F1": f1})
                if f1 > bf:
                    bf, bt = f1, th
            thr_best[str(cls)], f1s[cls] = bt, bf
        mf1 = (f1s[BC] + f1s[TC]) / 2
        print(f"  [tune] min_distance={md}: thr={thr_best} -> mF1 {mf1:.4f}")
        if best is None or mf1 > best[0]:
            best = (mf1, {"min_distance": md, "thr_fg": thr_fg, "thr": thr_best,
                          "smooth_sigma": smooth_sigma})
    return best[1], best[0], log


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def tissue_confusion(pred, gt, n=2, ignore=255):
    """pred, gt: (H, W) 0=BG, 1=CA (gt 의 255 는 제외)."""
    v = gt != ignore
    return np.bincount(gt[v].astype(np.int64) * n + pred[v].astype(np.int64), minlength=n * n).reshape(n, n)


def miou_from_confusion(cm):
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    return {"IoU/BG": float(iou[0]), "IoU/CA": float(iou[1]), "mIoU": float(np.nanmean(iou))}


def make_eval_loader(records, cfg, num_workers=4):
    norm = MacenkoNormalizer() if cfg["stain"] == "macenko" else None
    ds = OcelotEvalDataset(records, stain_normalizer=norm)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers,
                      collate_fn=lambda b: b[0])


def evaluate(model, records, cfg, post=None, tta=False, keep_probs=False, num_workers=4,
             tissue_out_dir=None):
    """records 전체 추론 + 공식 mF1 + tissue mIoU.
    반환: metrics dict, preds(이미지별 점 리스트), gts, (keep_probs 면) cell_probs 리스트"""
    post = post or DEFAULT_POST
    model.eval()
    preds, gts, probs = [], [], []
    cm = np.zeros((2, 2), np.int64)
    t0 = time.time()
    for s in make_eval_loader(records, cfg, num_workers):
        cp, tp = predict_pair(model, s["cell"], s["tissue"], s["box"], cfg["cell_target"], tta=tta,
                              amp=cfg.get("amp", True))
        cands = candidate_peaks(cp, cfg["cell_target"], post["min_distance"], post["thr_fg"],
                                post["smooth_sigma"])
        preds.append(filter_points(cands, post["thr"]))
        gts.append([tuple(int(v) for v in p) for p in s["points"]])
        tpred = tp.argmax(0).astype(np.uint8)
        cm += tissue_confusion(tpred, s["tissue_mask"])
        if tissue_out_dir:
            cv2.imwrite(os.path.join(tissue_out_dir, f"{s['sid']}.png"), tpred + 1)  # 1=BG, 2=CA
        if keep_probs:
            probs.append(cp.astype(np.float16))
    metrics = official_eval.score(preds, gts)
    metrics.update(miou_from_confusion(cm))
    metrics["eval_sec"] = round(time.time() - t0, 1)
    return metrics, preds, gts, probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_checkpoint_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg_build = dict(cfg)
    if cfg.get("norm", "auto") == "auto":     # 학습 때와 같은 정규화 값 유지
        lunit = cfg.get("encoder_ckpt") and "lunit" in str(cfg["encoder_ckpt"]).lower()
        cfg_build["norm"] = "lunit" if lunit else "imagenet"
    cfg_build["encoder_ckpt"] = None          # 학습된 가중치를 덮어쓸 것이므로 다시 다운로드 안 함
    cfg_build["encoder_weights"] = "none"
    model = build_model(cfg_build)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--data_root", default=None, help="생략하면 학습 때 경로")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--exp_name", default=None, help="생략하면 학습 때 exp_name")
    ap.add_argument("--tta", action="store_true", help="8방향 (flip x rot90) 평균")
    ap.add_argument("--tune", action="store_true", help="이 split 에서 후처리 파라미터 튜닝 (val 에서만!)")
    ap.add_argument("--postproc", default=None, help="후처리 json (기본: results/<exp>/postproc.json 있으면 사용)")
    ap.add_argument("--official_eval", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="앞의 N장만 (디버그)")
    args = ap.parse_args()

    if args.tune and args.split == "test":
        raise SystemExit("test 에서 튜닝하면 안 됩니다. val 에서 --tune 하세요.")
    official_eval.load(args.official_eval)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_checkpoint_model(args.ckpt, device)
    data_root = args.data_root or cfg["data_root"]
    exp = args.exp_name or cfg["exp_name"]
    out_dir = os.path.join(args.results_dir, exp)
    os.makedirs(out_dir, exist_ok=True)
    tissue_dir = os.path.join(out_dir, f"{args.split}_tissue")
    os.makedirs(tissue_dir, exist_ok=True)

    records = build_records(data_root, args.split, exclude_noisy=False)
    if args.limit:
        records = records[:args.limit]
    pp_path = args.postproc or os.path.join(out_dir, "postproc.json")
    post = DEFAULT_POST
    if not args.tune and os.path.exists(pp_path):
        with open(pp_path) as f:
            post = json.load(f)
        print(f"후처리 파라미터 로드: {pp_path} -> {post}")
    print(f"[{exp}] {args.split} {len(records)}장 추론 (epoch {ck.get('epoch')}, tta={args.tta})")

    metrics, preds, gts, probs = evaluate(model, records, cfg, post, tta=args.tta, keep_probs=args.tune,
                                          num_workers=args.num_workers, tissue_out_dir=tissue_dir)
    if args.tune:
        post, best_mf1, log = tune_postproc(probs, gts, cfg["cell_target"])
        with open(pp_path, "w") as f:
            json.dump(post, f, indent=2)
        with open(os.path.join(out_dir, "postproc_grid.json"), "w") as f:
            json.dump(log, f)
        print(f"튜닝 결과 저장: {pp_path} {post} (val mF1 {best_mf1:.4f})")
        cands = [candidate_peaks(p, cfg["cell_target"], post["min_distance"], post["thr_fg"],
                                 post["smooth_sigma"]) for p in probs]
        preds = [filter_points(c, post["thr"]) for c in cands]
        metrics.update(official_eval.score(preds, gts))

    pred_json = official_eval.to_json(preds)
    official_eval.check_validity(pred_json)
    with open(os.path.join(out_dir, f"{args.split}_pred.json"), "w") as f:
        json.dump(pred_json, f)
    with open(os.path.join(out_dir, f"gt_{args.split}.json"), "w") as f:
        json.dump(official_eval.to_json(gts), f)
    with open(os.path.join(out_dir, f"{args.split}_infer_info.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "epoch": ck.get("epoch"), "tta": args.tta, "post": post,
                   "data_root": data_root, "ids": [r.sid for r in records]}, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"저장: {out_dir}/{args.split}_pred.json  -> 상세 분석: "
          f"python -m src.analyze --exp {exp} --split {args.split}")


if __name__ == "__main__":
    main()
