"""빠른 동작 확인 (가짜 데이터 사용).

    python scripts/make_dummy_data.py --out data/dummy
    python -m pytest tests -q        # 또는  python tests/test_pipeline.py
"""
import os
import sys

import numpy as np
import torch
from torchvision.ops import roi_align

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.augment import TrainAugment, apply_dihedral, dihedral_points, random_crop_pair  # noqa: E402
from src.dataset import build_records, load_sample  # noqa: E402
from src.stain import MacenkoNormalizer, hed_jitter  # noqa: E402

DUMMY = os.environ.get("DUMMY_ROOT", "data/dummy")


def _roi(img, box, size):
    t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)[None]
    o = roi_align(t, torch.tensor([[0, *map(float, box)]]), (size, size), 1.0, 2, True)
    return o[0].permute(1, 2, 0).numpy()


def test_dihedral_points_match_image():
    """점 변환이 이미지 변환과 같은 픽셀을 가리키는지 (8가지 모두)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    pts = np.stack([rng.integers(0, 64, 20), rng.integers(0, 64, 20), np.ones(20, int)], 1)
    from src.augment import dihedral_image
    for k in range(4):
        for flip in (False, True):
            t = dihedral_image(img, k, flip)
            tp = dihedral_points(pts, 64, k, flip)
            assert np.array_equal(t[tp[:, 1], tp[:, 0]], img[pts[:, 1], pts[:, 0]]), (k, flip)


def test_box_follows_tissue_after_dihedral_and_crop():
    """dihedral + crop 후에도 roi_align(tissue, box) 가 cell 패치와 같은 장면인지."""
    rec = build_records(DUMMY, "train")[0]
    base = load_sample(rec)
    ref = _roi(base["tissue_img"], base["box"], 256)
    cell_small = base["cell_img"][::4, ::4].astype(np.float32)
    err0 = np.abs(ref - cell_small).mean()
    rng = np.random.default_rng(1)
    for k in range(4):
        for flip in (False, True):
            s = apply_dihedral({kk: (v.copy() if hasattr(v, "copy") else v) for kk, v in base.items()}, k, flip)
            s = random_crop_pair(s, 512, 512, rng)
            r = _roi(s["tissue_img"], s["box"], 128)
            c = s["cell_img"][::4, ::4].astype(np.float32)
            err = np.abs(r - c).mean()
            assert err < err0 * 1.5 + 1, (k, flip, err, err0)  # 틀린 box 면 ~4배 커짐


def test_train_augment_shapes():
    rec = build_records(DUMMY, "train")[0]
    for mode in ("none", "paired", "independent"):
        aug = TrainAugment(512, 512, stain_mode=mode)
        s = aug(load_sample(rec), np.random.default_rng(0))
        assert s["cell_img"].shape == (512, 512, 3) and s["tissue_img"].shape == (512, 512, 3)
        assert s["tissue_mask"].shape == (512, 512)
        assert (s["points"][:, :2] >= 0).all() and (s["points"][:, :2] < 512).all()
        x0, y0, x1, y1 = s["box"]
        assert abs((x1 - x0) - 128) < 1e-3 and 0 <= x0 and x1 <= 512 and 0 <= y0 and y1 <= 512


def test_stain_identity_and_macenko():
    rec = build_records(DUMMY, "train")[0]
    img = load_sample(rec)["cell_img"]
    same = hed_jitter(img, np.ones(3), np.zeros(3))
    assert np.abs(same.astype(int) - img).max() <= 1          # 반올림 오차만
    rnd = np.random.default_rng(0).integers(0, 256, (16, 16, 3)).astype(np.uint8)
    assert np.abs(hed_jitter(rnd, np.ones(3), np.zeros(3)).astype(int) - rnd).max() <= 1
    out = MacenkoNormalizer()(img)
    assert out.shape == img.shape and out.dtype == np.uint8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
