"""Tissue branch + Cell branch (DeepLabV3+) 와 tissue -> cell fusion.

구조 (Pred-to-input, OCELOT 논문)
    tissue 패치 --tissue_net--> tissue logits (2: BG/CA) --softmax-->
        cell 영역(box) crop + cell 해상도로 upsample (roi_align) = tissue prior (2ch)
    [cell RGB (3ch) , tissue prior (2ch)] --cell_net--> cell logits

Tissue random dropout (학습 때만)
    확률 p 로 샘플별 tissue prior 를 0(zero) 또는 균등분포 0.5(uniform) 로 바꿈.
    추론 때는 항상 tissue 정보 사용.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision.ops import roi_align

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Lunit benchmark-ssl-pathology README 의 정규화 값
LUNIT_MEAN = (0.70322989, 0.53606487, 0.66096631)
LUNIT_STD = (0.21716536, 0.26081574, 0.19938928)

LUNIT_URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
LUNIT_WEIGHTS = {
    "lunit_bt": "bt_rn50_ep200.torch",
    "lunit_mocov2": "mocov2_rn50_ep200.torch",
    "lunit_swav": "swav_rn50_ep200.torch",
}

NUM_TISSUE_CLASSES = 2   # BG, CA


def load_encoder_weights(encoder, src, verbose=True):
    """Lunit SSL ResNet-50 (torchvision 키 이름) 가중치를 smp encoder 에 로드.

    src: 'lunit_bt' | 'lunit_mocov2' | 'lunit_swav' | 로컬 파일 경로.
    in_channels 가 3 보다 크면 (예: cell branch 5ch) conv1 의 앞 3채널에 복사하고 나머지는 0.
    """
    if src in LUNIT_WEIGHTS:
        sd = torch.hub.load_state_dict_from_url(f"{LUNIT_URL_PREFIX}/{LUNIT_WEIGHTS[src]}",
                                                map_location="cpu", progress=True)
    else:
        sd = torch.load(src, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    clean = {}
    for k, v in sd.items():
        for pre in ("module.", "encoder.", "backbone.", "model."):
            if k.startswith(pre):
                k = k[len(pre):]
        clean[k] = v

    own = encoder.state_dict()
    new, skipped = {}, []
    for k, v in clean.items():
        if k not in own:
            skipped.append(k)
        elif own[k].shape == v.shape:
            new[k] = v
        elif k == "conv1.weight" and v.shape[1] == 3 and own[k].shape[1] > 3:
            w = torch.zeros_like(own[k])
            w[:, :3] = v
            new[k] = w
        else:
            skipped.append(k)
    missing = [k for k in own if k not in new]
    encoder.load_state_dict(new, strict=False)
    if verbose:
        print(f"[encoder weights] {src}: loaded {len(new)}/{len(own)} tensors, "
              f"missing {len(missing)} {missing[:5]}, skipped {len(skipped)} {skipped[:5]}")
    if len(new) < 0.9 * len(own):
        raise RuntimeError("encoder 가중치 키가 대부분 맞지 않습니다. 키 이름 매핑을 확인하세요.")


class OcelotNet(nn.Module):
    def __init__(self, encoder="resnet50", encoder_weights="imagenet", encoder_ckpt=None,
                 norm="auto", fusion="input", detach_tissue=True,
                 tissue_dropout=0.0, tissue_dropout_mode="zero", cell_out_channels=3):
        super().__init__()
        assert fusion in ("input", "none")
        assert tissue_dropout_mode in ("zero", "uniform")
        self.fusion = fusion
        self.detach_tissue = detach_tissue
        self.tissue_dropout = tissue_dropout
        self.tissue_dropout_mode = tissue_dropout_mode

        enc_w = None if (encoder_ckpt or encoder_weights in (None, "none")) else encoder_weights
        cell_in = 3 + (NUM_TISSUE_CLASSES if fusion == "input" else 0)
        self.tissue_net = smp.DeepLabV3Plus(encoder, encoder_weights=enc_w, in_channels=3,
                                            classes=NUM_TISSUE_CLASSES)
        self.cell_net = smp.DeepLabV3Plus(encoder, encoder_weights=enc_w, in_channels=cell_in,
                                          classes=cell_out_channels)
        if encoder_ckpt:
            load_encoder_weights(self.tissue_net.encoder, encoder_ckpt)
            load_encoder_weights(self.cell_net.encoder, encoder_ckpt)

        if norm == "auto":
            norm = "lunit" if (encoder_ckpt and "lunit" in str(encoder_ckpt).lower()) else "imagenet"
        mean, std = (LUNIT_MEAN, LUNIT_STD) if norm == "lunit" else (IMAGENET_MEAN, IMAGENET_STD)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def encoder_parameters(self):
        return list(self.tissue_net.encoder.parameters()) + list(self.cell_net.encoder.parameters())

    def _norm(self, x):
        return (x - self.mean) / self.std

    def tissue_prior(self, tissue_logits, box, out_size):
        """tissue 확률맵에서 cell 영역(box)을 잘라 cell 해상도로 키움. box: (B, 4) 연속 좌표."""
        prob = torch.softmax(tissue_logits.float(), dim=1)
        if self.detach_tissue:
            prob = prob.detach()
        idx = torch.arange(box.shape[0], device=box.device, dtype=torch.float32)[:, None]
        rois = torch.cat([idx, box.float()], dim=1)
        return roi_align(prob, rois, output_size=out_size, spatial_scale=1.0,
                         sampling_ratio=2, aligned=True)

    def apply_tissue_dropout(self, prior):
        if not self.training or self.tissue_dropout <= 0:
            return prior
        drop = torch.rand(prior.shape[0], 1, 1, 1, device=prior.device) < self.tissue_dropout
        fill = torch.zeros_like(prior) if self.tissue_dropout_mode == "zero" \
            else torch.full_like(prior, 1.0 / prior.shape[1])
        return torch.where(drop, fill, prior)

    def forward(self, cell, tissue, box):
        """cell, tissue: (B, 3, H, W) float [0, 1]. box: (B, 4).
        반환: cell_logits (B, C, Hc, Wc), tissue_logits (B, 2, Ht, Wt)."""
        tissue_logits = self.tissue_net(self._norm(tissue))
        x = self._norm(cell)
        if self.fusion == "input":
            prior = self.tissue_prior(tissue_logits, box, tuple(cell.shape[-2:]))
            prior = self.apply_tissue_dropout(prior)
            x = torch.cat([x, prior.to(x.dtype)], dim=1)
        cell_logits = self.cell_net(x)
        return cell_logits, tissue_logits


def build_model(cfg):
    return OcelotNet(
        encoder=cfg["encoder"], encoder_weights=cfg["encoder_weights"],
        encoder_ckpt=cfg.get("encoder_ckpt"), norm=cfg.get("norm", "auto"),
        fusion=cfg["fusion"], detach_tissue=cfg["detach_tissue"],
        tissue_dropout=cfg["tissue_dropout"], tissue_dropout_mode=cfg["tissue_dropout_mode"],
        cell_out_channels=3 if cfg["cell_target"] == "disk" else 2,
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def soft_dice_loss(logits, target, ignore_index=None, classes=None, eps=1.0):
    """multiclass soft Dice. target: (B, H, W) long. classes: Dice 를 계산할 클래스 목록."""
    prob = torch.softmax(logits.float(), dim=1)
    n = prob.shape[1]
    valid = torch.ones_like(target, dtype=torch.bool) if ignore_index is None else target != ignore_index
    t = torch.where(valid, target, torch.zeros_like(target))
    onehot = F.one_hot(t, n).permute(0, 3, 1, 2).float()
    v = valid[:, None].float()
    prob, onehot = prob * v, onehot * v
    inter = (prob * onehot).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    if classes is not None:
        dice = dice[classes]
    return 1 - dice.mean()


def masked_ce(logits, target, ignore_index=255):
    valid = (target != ignore_index).sum()
    if valid == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits.float(), target, ignore_index=ignore_index, reduction="sum") / valid


def focal_heatmap_loss(logits, gt, alpha=2, beta=4):
    """CenterNet 식 penalty-reduced focal loss. gt: (B, 2, H, W) gaussian heatmap (peak=1)."""
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = gt.ge(0.999).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - gt) ** beta * neg
    num_pos = pos.sum().clamp_min(1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def cell_loss(logits, target, mode):
    if mode == "disk":
        return F.cross_entropy(logits.float(), target) + soft_dice_loss(logits, target, classes=[1, 2])
    return focal_heatmap_loss(logits, target)


def tissue_loss(logits, mask):
    return masked_ce(logits, mask) + soft_dice_loss(logits, mask, ignore_index=255)
