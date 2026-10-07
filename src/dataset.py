"""OCELOT 데이터 로딩.

폴더 구조 (Zenodo ocelot2023_v1.0.x 압축 해제 기준):

    <data_root>/
    ├── metadata.json
    ├── images/{train,val,test}/{cell,tissue}/<id>.jpg
    └── annotations/{train,val,test}/
        ├── cell/<id>.csv      # 한 줄에 "x,y,class" (header 없음), class 1=BC, 2=TC
        └── tissue/<id>.png    # 1=BG, 2=CA, 255=UNK

좌표 규약
- cell 점 (x, y): 픽셀 index (0~1023). 공식 평가 코드와 같은 규약.
- cell box (tissue 패치 안에서 cell 패치가 차지하는 영역): 연속 좌표 (x0, y0, x1, y1).
  픽셀 i 는 [i, i+1) 구간을 차지한다고 본다 (torchvision roi_align(aligned=True) 와 같은 규약).

실행하면 데이터 구조 점검:
    python -m src.dataset --data_root data/ocelot2023_v1.0.1
"""
import argparse
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# cell 패치(0.2 MPP, 1024px) 는 tissue 패치(약 0.8 MPP, 1024px) 안에서 한 변 256px 영역
CELL_IN_TISSUE = 256

BC, TC = 1, 2                     # cell 클래스 (원본 CSV 값 그대로 사용)
CELL_CLASS_NAMES = {BC: "BC", TC: "TC"}
TISSUE_BG, TISSUE_CA = 0, 1       # 학습용 tissue 라벨 (원본: 1=BG, 2=CA)
IGNORE = 255                      # 원본 UNK(255) 및 그 밖의 값

# 팀원(민보성)이 찾은 train 라벨 불일치 샘플. --exclude_noisy 일 때만 사용, val/test 는 건드리지 않음.
NOISY_TISSUE_IDS = {242, 248, 313}   # tissue 라벨만 UNK 처리
NOISY_EXCLUDE_IDS = {152}            # 학습에서 제외

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass
class SampleRecord:
    sid: str                 # 파일 이름 stem (예: "001")
    split: str
    cell_img: str
    tissue_img: str
    cell_csv: str | None
    tissue_png: str | None
    box: tuple               # tissue 패치 좌표계에서 cell 패치 영역 (x0, y0, x1, y1)
    organ: str = "unknown"
    meta: dict = field(default_factory=dict)
    tissue_all_unk: bool = False   # --exclude_noisy 로 tissue 라벨을 전부 UNK 처리할 샘플


# ---------------------------------------------------------------------------
# 파일 읽기
# ---------------------------------------------------------------------------
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_cell_csv(path):
    """(N, 3) int64 배열 [x, y, class]. 빈 파일이면 (0, 3)."""
    if path is None or not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 3), np.int64)
    pts = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    return pts.reshape(-1, 3)


def read_tissue_mask(path):
    """원본 PNG(1=BG, 2=CA, 255=UNK) -> 학습용 (0=BG, 1=CA, 255=ignore)."""
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[..., 0]
    out = np.full(m.shape, IGNORE, np.uint8)
    out[m == 1] = TISSUE_BG
    out[m == 2] = TISSUE_CA
    return out


def load_metadata(data_root):
    """metadata.json 을 {int id: dict} 로 통일."""
    path = os.path.join(data_root, "metadata.json")
    with open(path) as f:
        meta = json.load(f)
    if isinstance(meta, dict) and "sample_pairs" in meta:
        meta = meta["sample_pairs"]
    if isinstance(meta, list):
        # 리스트 형식이면 id 가 1 부터 순서대로라고 가정
        return {i + 1: m for i, m in enumerate(meta)}
    return {int(k): v for k, v in meta.items()}


def cell_box_from_meta(m, tissue_size=1024):
    """metadata 의 patch_{x,y}_offset (tissue 패치 안 cell 패치 '중심'의 상대좌표) -> 연속좌표 box."""
    cx = float(m["patch_x_offset"]) * tissue_size
    cy = float(m["patch_y_offset"]) * tissue_size
    h = CELL_IN_TISSUE / 2
    return (cx - h, cy - h, cx + h, cy + h)


def _index_files(folder, exts):
    out = {}
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if Path(p).suffix.lower() in exts:
            out[Path(p).stem] = p
    return out


def build_records(data_root, split, exclude_noisy=False):
    """split 의 샘플 목록. 순서는 파일 이름 정렬 순 = 공식 GT json 의 image_{idx} 순서."""
    meta = load_metadata(data_root)
    cell_imgs = _index_files(os.path.join(data_root, "images", split, "cell"), IMG_EXTS)
    tissue_imgs = _index_files(os.path.join(data_root, "images", split, "tissue"), IMG_EXTS)
    cell_csvs = _index_files(os.path.join(data_root, "annotations", split, "cell"), (".csv",))
    tissue_pngs = _index_files(os.path.join(data_root, "annotations", split, "tissue"), (".png",))
    if not cell_imgs:
        raise FileNotFoundError(
            f"{data_root}/images/{split}/cell 에 이미지가 없습니다. --data_root 를 확인하세요.")

    records = []
    for sid in sorted(cell_imgs):
        if sid not in tissue_imgs:
            raise FileNotFoundError(f"tissue 이미지 없음: {split}/{sid}")
        try:
            iid = int(sid)
        except ValueError:
            iid = None
        if exclude_noisy and split == "train" and iid in NOISY_EXCLUDE_IDS:
            continue
        m = meta.get(iid, {}) if iid is not None else {}
        if "patch_x_offset" not in m:
            raise KeyError(f"metadata 에 {sid} 의 patch_x_offset 이 없습니다.")
        records.append(SampleRecord(
            sid=sid, split=split,
            cell_img=cell_imgs[sid], tissue_img=tissue_imgs[sid],
            cell_csv=cell_csvs.get(sid), tissue_png=tissue_pngs.get(sid),
            box=cell_box_from_meta(m), organ=m.get("organ", "unknown"), meta=m,
            tissue_all_unk=bool(exclude_noisy and split == "train" and iid in NOISY_TISSUE_IDS),
        ))
    return records


def load_sample(rec, stain_normalizer=None):
    """한 쌍을 numpy 로 읽기."""
    cell = read_rgb(rec.cell_img)
    tissue = read_rgb(rec.tissue_img)
    if stain_normalizer is not None:          # Macenko 정규화 (train/val/test 모두 적용)
        cell = stain_normalizer(cell)
        tissue = stain_normalizer(tissue)
    if rec.tissue_png is not None:
        mask = read_tissue_mask(rec.tissue_png)
    else:
        mask = np.full(tissue.shape[:2], IGNORE, np.uint8)
    if rec.tissue_all_unk:
        mask[:] = IGNORE
    return {
        "cell_img": cell,
        "tissue_img": tissue,
        "tissue_mask": mask,
        "points": read_cell_csv(rec.cell_csv),   # (N, 3) x, y, cls
        "box": np.asarray(rec.box, np.float32),
    }


# ---------------------------------------------------------------------------
# Cell 학습 타깃
# ---------------------------------------------------------------------------
def disk_target(points, size, radius=7):
    """(H, W) uint8 : 0=배경, 1=BC, 2=TC. 각 점에 반지름 radius 원판."""
    t = np.zeros((size, size), np.uint8)
    for x, y, c in points:
        cv2.circle(t, (int(x), int(y)), int(radius), int(c), -1)
    return t


def gaussian_target(points, size, sigma=6.0):
    """(2, H, W) float32 heatmap : 채널 0=BC, 1=TC. 점 위치 값이 1."""
    t = np.zeros((2, size, size), np.float32)
    r = int(3 * sigma)
    ax = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sigma ** 2))
    for x, y, c in points:
        x, y = int(x), int(y)
        x0, x1 = max(0, x - r), min(size, x + r + 1)
        y0, y1 = max(0, y - r), min(size, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        patch = g[y0 - (y - r):y1 - (y - r), x0 - (x - r):x1 - (x - r)]
        ch = int(c) - 1
        np.maximum(t[ch, y0:y1, x0:x1], patch, out=t[ch, y0:y1, x0:x1])
    return t


def to_chw_uint8(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class OcelotTrainDataset(Dataset):
    """학습용: 증강 + crop 후 텐서 반환.

    반환 dict
      cell      : (3, Cc, Cc) uint8
      tissue    : (3, Tc, Tc) uint8
      tissue_mask: (Tc, Tc) int64 (0/1/255)
      cell_target: disk -> (Cc, Cc) int64, gaussian -> (2, Cc, Cc) float32
      box       : (4,) float32  tissue crop 좌표계에서 cell crop 영역
    """

    def __init__(self, records, augment, cell_target="disk", disk_radius=7,
                 gauss_sigma=6.0, stain_normalizer=None, repeats=1):
        self.records = records
        self.augment = augment
        self.cell_target = cell_target
        self.disk_radius = disk_radius
        self.gauss_sigma = gauss_sigma
        self.stain_normalizer = stain_normalizer
        self.repeats = repeats

    def __len__(self):
        return len(self.records) * self.repeats

    def __getitem__(self, i):
        rec = self.records[i % len(self.records)]
        s = load_sample(rec, self.stain_normalizer)
        # 워커마다 torch 시드가 다르므로 그걸로 numpy rng 생성
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        s = self.augment(s, rng)
        size = s["cell_img"].shape[0]
        if self.cell_target == "disk":
            tgt = torch.from_numpy(disk_target(s["points"], size, self.disk_radius).astype(np.int64))
        else:
            tgt = torch.from_numpy(gaussian_target(s["points"], size, self.gauss_sigma))
        return {
            "cell": to_chw_uint8(s["cell_img"]),
            "tissue": to_chw_uint8(s["tissue_img"]),
            "tissue_mask": torch.from_numpy(s["tissue_mask"].astype(np.int64)),
            "cell_target": tgt,
            "box": torch.from_numpy(np.asarray(s["box"], np.float32)),
        }


class OcelotEvalDataset(Dataset):
    """평가/추론용: 원본 1024 해상도 그대로. batch_size=1 + collate_fn=lambda b: b[0] 로 사용."""

    def __init__(self, records, stain_normalizer=None):
        self.records = records
        self.stain_normalizer = stain_normalizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        s = load_sample(rec, self.stain_normalizer)
        s["cell"] = to_chw_uint8(s["cell_img"])
        s["tissue"] = to_chw_uint8(s["tissue_img"])
        s["index"] = i
        s["sid"] = rec.sid
        s["organ"] = rec.organ
        return s


# ---------------------------------------------------------------------------
# 데이터 점검
# ---------------------------------------------------------------------------
def inspect(data_root):
    meta = load_metadata(data_root)
    print(f"[metadata] {len(meta)} pairs. 예시:")
    k0 = sorted(meta)[0]
    print(json.dumps({k0: meta[k0]}, indent=2, ensure_ascii=False)[:1500])
    for split in ("train", "val", "test"):
        if not os.path.isdir(os.path.join(data_root, "images", split)):
            print(f"\n[{split}] 없음")
            continue
        recs = build_records(data_root, split)
        n_tc = n_bc = 0
        mask_vals = set()
        organs = {}
        sizes = set()
        for r in recs:
            pts = read_cell_csv(r.cell_csv)
            n_bc += int((pts[:, 2] == BC).sum())
            n_tc += int((pts[:, 2] == TC).sum())
            organs[r.organ] = organs.get(r.organ, 0) + 1
        for r in recs[:20]:
            if r.tissue_png:
                mask_vals |= set(np.unique(cv2.imread(r.tissue_png, cv2.IMREAD_UNCHANGED)).tolist())
            sizes.add(read_rgb(r.cell_img).shape)
            sizes.add(read_rgb(r.tissue_img).shape)
        missing = sum(r.cell_csv is None for r in recs), sum(r.tissue_png is None for r in recs)
        boxes = np.array([r.box for r in recs])
        print(f"\n[{split}] {len(recs)} pairs | TC {n_tc}, BC {n_bc} | organs {organs}")
        print(f"  이미지 shape(앞 20개): {sizes}")
        print(f"  tissue PNG 원본 값(앞 20개): {sorted(mask_vals)}  (기대: 1=BG, 2=CA, 255=UNK)")
        print(f"  라벨 파일 없음 (cell csv, tissue png): {missing}")
        print(f"  cell box x0 범위 {boxes[:, 0].min():.1f}~{boxes[:, 0].max():.1f}, "
              f"y0 범위 {boxes[:, 1].min():.1f}~{boxes[:, 1].max():.1f} (0~768 이어야 정상)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    inspect(ap.parse_args().data_root)
