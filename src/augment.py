"""cell/tissue 쌍에 같은 기하 변환 + crop + (stain) + 약한 photometric 증강.

핵심 함정: tissue 패치를 뒤집거나 돌리면, tissue 안에서 cell 패치가 있는 위치(box)도
같이 변환해야 fusion 이 맞는 영역을 가리킨다. 여기서는 cell 패치, cell 점, tissue 패치,
tissue 마스크, cell box 다섯 개에 '같은' dihedral 변환(flip + rot90)을 적용한다.

dihedral 변환 정의 (k, flip):  먼저 좌우 flip(선택) -> 그다음 np.rot90(k) (반시계 90° k번).
연속 좌표(u in [0, S])에서
    좌우 flip : x' = S - x
    rot90 1회 : (x, y) -> (y, S - x)
점(픽셀 index)은 +0.5 해서 연속 좌표로 바꾼 뒤 변환하고 다시 -0.5.
"""
import cv2
import numpy as np

from .stain import StainAugment


# ---------------------------------------------------------------------------
# Dihedral (flip / rot90) helpers
# ---------------------------------------------------------------------------
def dihedral_image(img, k, flip):
    """(H, W[, C]) numpy 배열."""
    if flip:
        img = img[:, ::-1]
    return np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))


def dihedral_coords(xy, size, k, flip):
    """연속 좌표 (N, 2) 변환."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2).copy()
    if flip:
        xy[:, 0] = size - xy[:, 0]
    for _ in range(k % 4):
        xy = np.stack([xy[:, 1], size - xy[:, 0]], axis=1)
    return xy


def dihedral_points(points, size, k, flip):
    """(N, 3) [x, y, cls] 픽셀 index 점."""
    if len(points) == 0:
        return points
    out = points.copy()
    xy = dihedral_coords(points[:, :2] + 0.5, size, k, flip) - 0.5
    out[:, :2] = np.rint(xy).astype(points.dtype)
    return out


def dihedral_box(box, size, k, flip):
    """연속 좌표 box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    c = dihedral_coords([[x0, y0], [x1, y1]], size, k, flip)
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()], np.float32)


def apply_dihedral(s, k, flip):
    cs = s["cell_img"].shape[0]
    ts = s["tissue_img"].shape[0]
    assert s["cell_img"].shape[0] == s["cell_img"].shape[1], "정사각 패치만 지원"
    assert s["tissue_img"].shape[0] == s["tissue_img"].shape[1], "정사각 패치만 지원"
    s["cell_img"] = dihedral_image(s["cell_img"], k, flip)
    s["tissue_img"] = dihedral_image(s["tissue_img"], k, flip)
    s["tissue_mask"] = dihedral_image(s["tissue_mask"], k, flip)
    s["points"] = dihedral_points(s["points"], cs, k, flip)
    s["box"] = dihedral_box(s["box"], ts, k, flip)
    return s


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------
def random_crop_pair(s, cell_crop, tissue_crop, rng):
    """cell 패치에서 cell_crop 을 자르고, 그 영역을 포함하는 tissue_crop 을 tissue 에서 자른다.
    결과 box = tissue crop 좌표계에서 'cell crop' 이 차지하는 영역."""
    H = s["cell_img"].shape[0]
    T = s["tissue_img"].shape[0]
    bx0, by0, bx1, by1 = [float(v) for v in s["box"]]
    scale = (bx1 - bx0) / H                    # cell 1px 이 tissue 에서 몇 px 인지 (= 0.25)

    # 1) cell crop
    cx0 = cy0 = 0
    if cell_crop < H:
        cx0 = int(rng.integers(0, H - cell_crop + 1))
        cy0 = int(rng.integers(0, H - cell_crop + 1))
        s["cell_img"] = s["cell_img"][cy0:cy0 + cell_crop, cx0:cx0 + cell_crop]
        p = s["points"]
        if len(p):
            p = p.copy()
            p[:, 0] -= cx0
            p[:, 1] -= cy0
            keep = (p[:, 0] >= 0) & (p[:, 0] < cell_crop) & (p[:, 1] >= 0) & (p[:, 1] < cell_crop)
            s["points"] = p[keep]
    cs = min(cell_crop, H)
    region = np.array([bx0 + cx0 * scale, by0 + cy0 * scale,
                       bx0 + (cx0 + cs) * scale, by0 + (cy0 + cs) * scale], np.float64)

    # 2) region 을 포함하는 tissue crop
    tx0 = ty0 = 0
    if tissue_crop < T:
        def pick(lo_edge, hi_edge):
            lo = int(np.ceil(hi_edge - tissue_crop))
            hi = int(np.floor(lo_edge))
            lo, hi = max(lo, 0), min(hi, T - tissue_crop)
            if lo > hi:     # 포함 불가능하면 region 중심 기준으로 clamp
                c = int(round((lo_edge + hi_edge) / 2 - tissue_crop / 2))
                return int(np.clip(c, 0, T - tissue_crop))
            return int(rng.integers(lo, hi + 1))
        tx0 = pick(region[0], region[2])
        ty0 = pick(region[1], region[3])
        s["tissue_img"] = s["tissue_img"][ty0:ty0 + tissue_crop, tx0:tx0 + tissue_crop]
        s["tissue_mask"] = s["tissue_mask"][ty0:ty0 + tissue_crop, tx0:tx0 + tissue_crop]
    region -= [tx0, ty0, tx0, ty0]
    s["box"] = region.astype(np.float32)
    return s


# ---------------------------------------------------------------------------
# 약한 photometric
# ---------------------------------------------------------------------------
def color_jitter(img, b, c, sat):
    x = img.astype(np.float32)
    x = x * b                                   # brightness
    mean = x.mean()
    x = (x - mean) * c + mean                   # contrast
    gray = x.mean(axis=2, keepdims=True)
    x = (x - gray) * sat + gray                 # saturation
    return np.clip(x, 0, 255).astype(np.uint8)


def blur_noise(img, rng, p_blur=0.2, p_noise=0.2):
    if rng.random() < p_blur:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 1.0)))
    if rng.random() < p_noise:
        n = rng.normal(0, rng.uniform(1, 5), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# 전체 학습 증강
# ---------------------------------------------------------------------------
class TrainAugment:
    """순서: dihedral(같은 변환) -> crop -> stain 증강 -> 약한 photometric.

    stain_mode: none | paired | independent | macenko
      (macenko 는 데이터 로딩 단계의 정규화라 여기서는 아무것도 안 함)
    """

    def __init__(self, cell_crop=512, tissue_crop=512, geometric=True, photometric=True,
                 stain_mode="none", stain_alpha=0.05, stain_beta=0.01, jitter=0.1):
        assert tissue_crop >= cell_crop / 4, "tissue_crop 은 cell_crop/4 이상이어야 cell 영역을 포함"
        self.cell_crop = cell_crop
        self.tissue_crop = tissue_crop
        self.geometric = geometric
        self.photometric = photometric
        self.jitter = jitter
        self.stain = StainAugment(stain_mode, stain_alpha, stain_beta)

    def __call__(self, s, rng):
        if self.geometric:
            s = apply_dihedral(s, int(rng.integers(4)), bool(rng.random() < 0.5))
        s = random_crop_pair(s, self.cell_crop, self.tissue_crop, rng)
        s["cell_img"], s["tissue_img"] = self.stain(s["cell_img"], s["tissue_img"], rng)
        if self.photometric:
            j = self.jitter
            b, c, sat = rng.uniform(1 - j, 1 + j, 3)
            # 색 jitter 는 같은 슬라이드라 cell/tissue 같은 값, blur/noise 는 각각
            s["cell_img"] = blur_noise(color_jitter(s["cell_img"], b, c, sat), rng)
            s["tissue_img"] = blur_noise(color_jitter(s["tissue_img"], b, c, sat), rng)
        return s
