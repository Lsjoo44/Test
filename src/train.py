"""학습 스크립트. 실험은 configs/*.yaml 또는 CLI 플래그로 전환 (CLI 가 yaml 보다 우선).

    python -m src.train --config configs/E0.yaml
    python -m src.train --config configs/S1.yaml --epochs 5          # 일부 값만 덮어쓰기
    python -m src.train --config configs/E0.yaml --resume auto        # (기본) 중단된 학습 이어서

- checkpoints/<exp>/last.pth 를 매 epoch 저장 -> VESSL 24시간 종료/중단 후 같은 명령으로 재개
- checkpoints/<exp>/best.pth : val mF1 최고 (모델 선택은 val 기준만, test 는 맨 마지막 1회)
- checkpoints/<exp>/log.csv  : epoch 별 loss / val 지표 / 시간
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import official_eval
from .augment import TrainAugment
from .dataset import OcelotTrainDataset, build_records
from .infer import evaluate
from .model import build_model, cell_loss, tissue_loss
from .stain import MacenkoNormalizer


def str2none(v):
    return None if v in (None, "", "none", "None") else v


def get_cfg(argv=None):
    p = argparse.ArgumentParser(description="OCELOT cell/tissue multi-task 학습")
    B = argparse.BooleanOptionalAction
    p.add_argument("--config", default=None)
    # 경로
    p.add_argument("--data_root", default="data/ocelot2023_v1.0.1")
    p.add_argument("--exp_name", default="E0")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--official_eval", default=None, help="ocelot23algo/evaluation/eval.py 경로")
    # 모델
    p.add_argument("--encoder", default="resnet50")
    p.add_argument("--encoder_weights", default="imagenet", help="imagenet | none")
    p.add_argument("--encoder_ckpt", type=str2none, default=None,
                   help="lunit_bt | lunit_mocov2 | lunit_swav | 파일경로 (지정 시 encoder_weights 무시)")
    p.add_argument("--norm", default="auto", choices=["auto", "imagenet", "lunit"])
    p.add_argument("--fusion", default="input", choices=["input", "none"])
    p.add_argument("--detach_tissue", action=B, default=True,
                   help="fusion 에 넣는 tissue 확률을 detach (cell loss 가 tissue branch 로 안 흐름)")
    p.add_argument("--tissue_dropout", type=float, default=0.0)
    p.add_argument("--tissue_dropout_mode", default="zero", choices=["zero", "uniform"])
    # cell 타깃
    p.add_argument("--cell_target", default="disk", choices=["disk", "gaussian"])
    p.add_argument("--disk_radius", type=int, default=7)
    p.add_argument("--gauss_sigma", type=float, default=6.0)
    # 증강
    p.add_argument("--geometric", action=B, default=True)
    p.add_argument("--photometric", action=B, default=True)
    p.add_argument("--jitter", type=float, default=0.1, help="약한 brightness/contrast/saturation 폭")
    p.add_argument("--stain", default="none", choices=["none", "paired", "independent", "macenko"])
    p.add_argument("--stain_alpha", type=float, default=0.05)
    p.add_argument("--stain_beta", type=float, default=0.01)
    p.add_argument("--exclude_noisy", action=B, default=False, help="train 라벨 불일치 샘플 처리")
    # 학습
    p.add_argument("--cell_crop", type=int, default=512)
    p.add_argument("--tissue_crop", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--repeats", type=int, default=1, help="한 epoch 에 각 샘플을 몇 번 (랜덤 crop) 볼지")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cell_lr", type=float, default=None, help="cell branch lr (기본 lr)")
    p.add_argument("--tissue_lr", type=float, default=None, help="tissue branch lr (기본 lr)")
    p.add_argument("--encoder_lr_scale", type=float, default=1.0, help="encoder lr = branch lr x 이 값")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--lambda_tissue", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action=B, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_every", type=int, default=2)
    p.add_argument("--train_limit", type=int, default=0, help="디버그: train 앞 N장만")
    p.add_argument("--val_limit", type=int, default=0, help="디버그: val 앞 N장만")
    p.add_argument("--resume", default="auto", help="auto | none | ckpt 경로")

    args, _ = p.parse_known_args(argv)
    if args.config:
        with open(args.config) as f:
            ycfg = yaml.safe_load(f) or {}
        unknown = set(ycfg) - {a.dest for a in p._actions}
        if unknown:
            raise SystemExit(f"config 에 모르는 키: {unknown}")
        p.set_defaults(**ycfg)
    return vars(p.parse_args(argv))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, cfg):
    cell_lr = cfg["cell_lr"] or cfg["lr"]
    tissue_lr = cfg["tissue_lr"] or cfg["lr"]
    s = cfg["encoder_lr_scale"]
    groups = []
    for net, lr in ((model.cell_net, cell_lr), (model.tissue_net, tissue_lr)):
        enc = list(net.encoder.parameters())
        enc_ids = {id(q) for q in enc}
        rest = [q for q in net.parameters() if id(q) not in enc_ids]
        groups.append({"params": enc, "lr": lr * s})
        groups.append({"params": rest, "lr": lr})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def make_scheduler(opt, cfg, steps_per_epoch):
    total = cfg["epochs"] * steps_per_epoch
    warm = int(cfg["warmup_epochs"] * steps_per_epoch)

    def f(step):  # linear warmup -> cosine annealing
        if step < warm:
            return (step + 1) / max(warm, 1)
        t = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def append_log(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    cfg = get_cfg(argv)
    seed_all(cfg["seed"])
    official_eval.load(cfg["official_eval"])     # 시작 전에 공식 평가 코드 존재 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out = os.path.join(cfg["ckpt_dir"], cfg["exp_name"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # ---- data
    train_recs = build_records(cfg["data_root"], "train", exclude_noisy=cfg["exclude_noisy"])
    val_recs = build_records(cfg["data_root"], "val")
    if cfg["train_limit"]:
        train_recs = train_recs[:cfg["train_limit"]]
    if cfg["val_limit"]:
        val_recs = val_recs[:cfg["val_limit"]]
    aug = TrainAugment(cfg["cell_crop"], cfg["tissue_crop"], cfg["geometric"], cfg["photometric"],
                       cfg["stain"], cfg["stain_alpha"], cfg["stain_beta"], cfg["jitter"])
    norm = MacenkoNormalizer() if cfg["stain"] == "macenko" else None
    train_ds = OcelotTrainDataset(train_recs, aug, cfg["cell_target"], cfg["disk_radius"],
                                  cfg["gauss_sigma"], stain_normalizer=norm, repeats=cfg["repeats"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                        persistent_workers=cfg["num_workers"] > 0)
    if len(loader) == 0:
        raise SystemExit("train 샘플 수가 batch_size 보다 적습니다.")
    print(f"train {len(train_recs)}장 x{cfg['repeats']} -> {len(loader)} it/epoch | val {len(val_recs)}장 | {device}")

    # ---- model / optim
    model = build_model(cfg).to(device)
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg, len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")

    start_epoch, best, best_epoch = 0, -1.0, -1
    resume = cfg["resume"]
    if resume == "auto":
        resume = os.path.join(out, "last.pth") if os.path.exists(os.path.join(out, "last.pth")) else None
    elif resume == "none":
        resume = None
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, best_epoch = ck["epoch"] + 1, ck["best"], ck["best_epoch"]
        print(f"resume: {resume} -> epoch {start_epoch} 부터 (best mF1 {best:.4f} @ {best_epoch})")

    log_path = os.path.join(out, "log.csv")
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "cell": 0.0, "tissue": 0.0}
        for it, b in enumerate(loader):
            cell = b["cell"].to(device, non_blocking=True).float() / 255
            tissue = b["tissue"].to(device, non_blocking=True).float() / 255
            box = b["box"].to(device)
            tgt = b["cell_target"].to(device, non_blocking=True)
            mask = b["tissue_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=cfg["amp"] and device.type == "cuda"):
                cl, tl = model(cell, tissue, box)
            lc = cell_loss(cl, tgt, cfg["cell_target"])
            lt = tissue_loss(tl, mask)
            loss = lc + cfg["lambda_tissue"] * lt
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["grad_clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums["loss"] += loss.item()
            sums["cell"] += lc.item()
            sums["tissue"] += lt.item()
            if it % 20 == 0:
                print(f"  ep {epoch} it {it}/{len(loader)} loss {loss.item():.4f} "
                      f"(cell {lc.item():.4f}, tissue {lt.item():.4f})", flush=True)
        n = len(loader)
        row = {"epoch": epoch, "lr": opt.param_groups[1]["lr"],
               **{f"train_{k}": round(v / n, 5) for k, v in sums.items()},
               "train_sec": round(time.time() - t0, 1)}

        do_val = (epoch + 1) % cfg["val_every"] == 0 or epoch + 1 == cfg["epochs"]
        if do_val:
            m, *_ = evaluate(model, val_recs, cfg, num_workers=min(4, cfg["num_workers"]))
            row.update({f"val_{k}": v for k, v in m.items()})
            if m["mF1"] > best:
                best, best_epoch = m["mF1"], epoch
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val": m},
                           os.path.join(out, "best.pth"))
        row["best_mF1"], row["best_epoch"] = best, best_epoch
        append_log(log_path, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "cfg": cfg, "epoch": epoch,
                    "best": best, "best_epoch": best_epoch},
                   os.path.join(out, "last.pth.tmp"))
        os.replace(os.path.join(out, "last.pth.tmp"), os.path.join(out, "last.pth"))

    print(f"완료. best val mF1 {best:.4f} @ epoch {best_epoch} -> {out}/best.pth")


if __name__ == "__main__":
    main()
