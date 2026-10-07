"""H&E 염색 증강 / 정규화.

- HED jitter : rgb2hed 로 Hematoxylin/Eosin/DAB 농도 공간으로 바꾼 뒤
               각 채널에 alpha ~ U[1-σa, 1+σa] 곱하고 beta ~ U[-σb, σb] 더한 뒤 hed2rgb.
- paired     : cell/tissue 쌍에 같은 (alpha, beta)  -> 같은 슬라이드라 현실적
- independent: cell/tissue 에 각각 다른 (alpha, beta)
- macenko    : 증강이 아니라 '정규화'(색 통일). train/val/test 모두 같은 기준 색으로 변환.
"""
import numpy as np
from skimage.color import hed2rgb, rgb2hed


# ---------------------------------------------------------------------------
# HED jitter
# ---------------------------------------------------------------------------
def sample_hed_params(rng, sigma_alpha, sigma_beta):
    alpha = rng.uniform(1 - sigma_alpha, 1 + sigma_alpha, 3)
    beta = rng.uniform(-sigma_beta, sigma_beta, 3)
    return alpha, beta


def hed_jitter(img, alpha, beta):
    """img: (H, W, 3) uint8 -> uint8."""
    hed = rgb2hed(img)
    hed = hed * alpha + beta
    rgb = hed2rgb(hed)
    return np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)


class StainAugment:
    MODES = ("none", "paired", "independent", "macenko")

    def __init__(self, mode="none", sigma_alpha=0.05, sigma_beta=0.01):
        assert mode in self.MODES, f"stain mode 는 {self.MODES} 중 하나"
        self.mode = mode
        self.sa = sigma_alpha
        self.sb = sigma_beta

    def __call__(self, cell, tissue, rng):
        if self.mode == "paired":
            a, b = sample_hed_params(rng, self.sa, self.sb)
            return hed_jitter(cell, a, b), hed_jitter(tissue, a, b)
        if self.mode == "independent":
            a1, b1 = sample_hed_params(rng, self.sa, self.sb)
            a2, b2 = sample_hed_params(rng, self.sa, self.sb)
            return hed_jitter(cell, a1, b1), hed_jitter(tissue, a2, b2)
        return cell, tissue


# ---------------------------------------------------------------------------
# Macenko normalization (torchstain numpy 구현과 같은 알고리즘)
# ---------------------------------------------------------------------------
class MacenkoNormalizer:
    """기준 색(HERef, maxCRef)으로 변환. target 이미지를 fit 하지 않으면 torchstain 기본값 사용."""

    def __init__(self, Io=240, alpha=1, beta=0.15, max_fit_pixels=200_000, seed=0):
        self.Io, self.alpha, self.beta = Io, alpha, beta
        self.max_fit_pixels = max_fit_pixels
        self.rng = np.random.default_rng(seed)
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def _od(self, I):
        return -np.log((I.astype(np.float64) + 1) / self.Io)

    def _stain_matrix(self, OD):
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        if len(ODhat) > self.max_fit_pixels:   # 속도: 일부 픽셀로만 추정
            ODhat = ODhat[self.rng.choice(len(ODhat), self.max_fit_pixels, replace=False)]
        if len(ODhat) < 100:
            raise ValueError("조직 픽셀이 너무 적음")
        _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        eigvecs = eigvecs[:, [1, 2]]
        That = ODhat @ eigvecs
        phi = np.arctan2(That[:, 1], That[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)
        v_min = eigvecs @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v_max = eigvecs @ np.array([np.cos(max_phi), np.sin(max_phi)])
        # H 를 첫 번째 열로
        HE = np.stack([v_min, v_max], 1) if v_min[0] > v_max[0] else np.stack([v_max, v_min], 1)
        return HE

    def _concentrations(self, OD, HE):
        C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        maxC = np.array([np.percentile(C[0], 99), np.percentile(C[1], 99)])
        return C, maxC

    def fit(self, target_rgb):
        OD = self._od(target_rgb.reshape(-1, 3))
        HE = self._stain_matrix(OD)
        _, maxC = self._concentrations(OD, HE)
        self.HERef, self.maxCRef = HE, maxC
        return self

    def __call__(self, img):
        """uint8 (H, W, 3) -> uint8. 실패(배경만 있는 패치 등)하면 원본 반환."""
        h, w, _ = img.shape
        try:
            OD = self._od(img.reshape(-1, 3))
            HE = self._stain_matrix(OD)
            C, maxC = self._concentrations(OD, HE)
            C *= (self.maxCRef / np.maximum(maxC, 1e-6))[:, None]
            out = self.Io * np.exp(-self.HERef @ C)
        except (ValueError, np.linalg.LinAlgError):
            return img
        return np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)
