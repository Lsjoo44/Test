"""공식 평가 코드 (lunit-io/ocelot23algo/evaluation/eval.py) 를 그대로 불러와 쓰는 얇은 래퍼.

mF1 계산 로직은 직접 구현하지 않고, 공식 파일의 함수
(_check_validity, _preprocess_distance_and_confidence, _calc_scores, DISTANCE_CUTOFF,
 CLS_IDX_TO_NAME) 를 import 해서 호출한다. (공식 main() 은 json 을 두 번 읽는 버그가 있어 쓰지 않음)

eval.py 위치 우선순위: 인자 -> 환경변수 OCELOT_EVAL -> <repo>/third_party/ocelot23algo/evaluation/eval.py
설치: bash scripts/setup.sh  (또는 git clone https://github.com/lunit-io/ocelot23algo third_party/ocelot23algo)
"""
import importlib.util
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = REPO_ROOT / "third_party" / "ocelot23algo" / "evaluation" / "eval.py"

_MODULE = None


def load(path=None):
    global _MODULE
    if _MODULE is not None and path is None:
        return _MODULE
    candidates = [path, os.environ.get("OCELOT_EVAL"), DEFAULT_EVAL]
    for p in candidates:
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("ocelot_official_eval", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _MODULE = mod
            return mod
    raise FileNotFoundError(
        "공식 평가 코드 eval.py 를 찾지 못했습니다. `bash scripts/setup.sh` 로 "
        "third_party/ocelot23algo 를 받거나 --official_eval / OCELOT_EVAL 로 경로를 지정하세요.")


def score(pred_per_image, gt_per_image, classes=None):
    """pred_per_image: 이미지별 [(x, y, cls, prob), ...]
    gt_per_image  : 이미지별 [(x, y, cls[, 1.0]), ...]
    classes       : 계산할 클래스 idx (기본: 공식 CLS_IDX_TO_NAME 전체)
    반환: {"Pre/BC", "Rec/BC", "F1/BC", ..., "mF1"}  (mF1 은 전체 클래스를 계산할 때만)"""
    E = load()
    gt = [[(int(g[0]), int(g[1]), int(g[2]), 1.0) for g in img] for img in gt_per_image]
    res = E._preprocess_distance_and_confidence(pred_per_image, gt)
    out = {}
    cls_items = [(c, n) for c, n in E.CLS_IDX_TO_NAME.items() if classes is None or c in classes]
    for cls_idx, name in cls_items:
        p, r, f = E._calc_scores(res, cls_idx, E.DISTANCE_CUTOFF)
        out[f"Pre/{name}"], out[f"Rec/{name}"], out[f"F1/{name}"] = p, r, f
    if classes is None:
        out["mF1"] = sum(out[f"F1/{n}"] for n in E.CLS_IDX_TO_NAME.values()) / len(E.CLS_IDX_TO_NAME)
    return out


def to_json(points_per_image):
    """공식 'Multiple points' 형식 dict. name = image_{idx} (idx = split 내 정렬 순서)."""
    pts = []
    for idx, img in enumerate(points_per_image):
        for p in img:
            pts.append({"name": f"image_{idx}",
                        "point": [int(p[0]), int(p[1]), int(p[2])],
                        "probability": float(p[3]) if len(p) > 3 else 1.0})
    return {"type": "Multiple points", "num_images": len(points_per_image),
            "points": pts, "version": {"major": 1, "minor": 0}}


def from_json(path):
    with open(path) as f:
        d = json.load(f)
    n = d.get("num_images")
    if n is None:
        n = 1 + max((int(p["name"].split("_")[-1]) for p in d["points"]), default=-1)
    out = [[] for _ in range(n)]
    for p in d["points"]:
        x, y, c = p["point"]
        out[int(p["name"].split("_")[-1])].append((x, y, c, p["probability"]))
    return out


def check_validity(json_dict):
    load()._check_validity(json_dict["points"])
