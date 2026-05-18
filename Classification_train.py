import os
import random
import time
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split

def add_noise(x, std=0.02):
    return x + std * torch.randn_like(x)

def clamp_01(x):
    return x.clamp(0.0, 1.0)

# Config

@dataclass
class CFG:
    project_root: Path = Path(__file__).resolve().parent
    data_root: Path = project_root / "Dataset"

    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 6

    epochs_renders: int = 6
    epochs_photos_unfreeze: int = 7

    lr_head: float = 1e-3
    lr_backbone: float = 3e-5

    val_split: float = 0.05
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir: Path = Path("checkpoints")


cfg = CFG()


# Utils
def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False



# Data
def make_render_loaders():

    render_tf = transforms.Compose([

        transforms.Resize((cfg.img_size, cfg.img_size)),

        transforms.RandomRotation(10),

        transforms.RandomHorizontalFlip(p=0.3),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.02
        ),

        transforms.RandomApply([
            transforms.GaussianBlur(3, (0.1, 1.0))
        ], p=0.2),

        transforms.ToTensor(),

        transforms.Lambda(add_noise),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])
    eval_tf = transforms.Compose([

        transforms.Resize((cfg.img_size, cfg.img_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    print("[INFO] Loading render dataset...")

    full_ds = datasets.ImageFolder(
        cfg.data_root / "renders",
        transform=render_tf
    )

    eval_ds = datasets.ImageFolder(
        cfg.data_root / "renders",
        transform=eval_tf
    )

    n_total = len(full_ds)
    n_val = int(n_total * cfg.val_split)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(cfg.seed)

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=g
    )

    val_ds.dataset = eval_ds

    loader_args = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    train_loader = DataLoader(train_ds, **loader_args)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader, len(full_ds.classes)


def make_photo_loaders():

    photo_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(3, (0.1, 1.5))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])
    eval_tf = transforms.Compose([

        transforms.Resize((cfg.img_size, cfg.img_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    print("[INFO] Loading photo dataset...")

    full_ds = datasets.ImageFolder(
        cfg.data_root / "photos",
        transform=photo_tf
    )

    targets = [y for _, y in full_ds.samples]
    class_counts = np.bincount(targets)

    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    eval_ds = datasets.ImageFolder(
        cfg.data_root / "photos",
        transform=eval_tf
    )

    n_total = len(full_ds)
    n_val = int(n_total * cfg.val_split)
    n_train = n_total - n_val

    
    #stratified split to maintain class distribution in train and val sets
    targets = np.array([y for _, y in full_ds.samples])
    indices = np.arange(len(targets))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg.val_split,
        random_state=cfg.seed,
        stratify=targets
    )

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(eval_ds, val_idx)
    print("Train class distribution:", np.bincount(targets[train_idx]))
    print("Val class distribution:", np.bincount(targets[val_idx]))

    loader_args = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    train_loader = DataLoader(train_ds, **loader_args)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, val_loader, class_weights



# Model

def build_model(num_classes: int):

    # model = models.resnet18(weights="IMAGENET1K_V1")
    model = models.resnet50(weights="IMAGENET1K_V2")

    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)

    return model



# Train / Eval

def train_one_epoch(model, loader, optimizer, device, scaler, epoch, stage, class_weights=None):


    model.train()
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total_acc = 0
    total = 0

    start_time = time.time()

    print(f"\n[INFO] {stage} | Epoch {epoch+1} started...")

    for i, (x, y) in enumerate(loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Debug label range
        if y.min() < 0 or y.max() >= model.fc.out_features:
            print("\n[ERROR] Invalid label detected!")
            print("Min label:", y.min().item())
            print("Max label:", y.max().item())
            print("Num classes:", model.fc.out_features)
            print("Batch index:", i)
            raise RuntimeError("Invalid label detected")

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out = model(x)
            loss = loss_fn(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)

        total_loss += loss.item() * bs
        total_acc += (out.argmax(1) == y).sum().item()
        total += bs

        # Progress print every 200 batches
        if (i + 1) % 200 == 0 or (i + 1) == len(loader):

            elapsed = time.time() - start_time
            done = i + 1
            total_batches = len(loader)

            batches_per_sec = done / elapsed
            eta = (total_batches - done) / batches_per_sec / 60

            avg_loss = total_loss / total
            avg_acc = total_acc / total

            print(
                f"[{stage}] "
                f"Epoch {epoch+1} "
                f"[{done}/{total_batches}] "
                f"Loss: {avg_loss:.4f} "
                f"Acc: {avg_acc:.4f} "
                f"ETA: {eta:.1f} min"
            )

    return total_loss / total, total_acc / total



@torch.no_grad()
def eval_one_epoch(model, loader, device, epoch):

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    total_acc = 0
    total = 0

    start_time = time.time()

    print(f"\n[INFO] Validation | Epoch {epoch+1} started...")

    for i, (x, y) in enumerate(loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)
        loss = loss_fn(out, y)

        bs = x.size(0)

        total_loss += loss.item() * bs
        total_acc += (out.argmax(1) == y).sum().item()
        total += bs

        if (i + 1) % 200 == 0 or (i + 1) == len(loader):

            elapsed = time.time() - start_time
            done = i + 1
            total_batches = len(loader)

            batches_per_sec = done / elapsed
            eta = (total_batches - done) / batches_per_sec / 60

            avg_loss = total_loss / total
            avg_acc = total_acc / total

            print(
                f"[VAL] "
                f"Epoch {epoch+1} "
                f"[{done}/{total_batches}] "
                f"Loss: {avg_loss:.4f} "
                f"Acc: {avg_acc:.4f} "
                f"ETA: {eta:.1f} min"
            )

    return total_loss / total, total_acc / total



# Main
def main():

    cfg.save_dir.mkdir(exist_ok=True)
    print("Dataset root:", cfg.data_root.resolve())
    print("Exists:", cfg.data_root.exists())

    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    render_train_loader, render_val_loader, num_classes = make_render_loaders()
    train_loader, val_loader, class_weights = make_photo_loaders()  
    class_weights = class_weights.to(device)

    model = build_model(num_classes)
    model.to(device)

    history = {
        "renders": {
            "train_loss": [],
            "train_acc": [],
            "val_render_loss": [],
            "val_render_acc": [],
            "val_real_loss": [],
            "val_real_acc": [],
        },
        "photos": {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
    }
    
    #mapping check
    render_ds = datasets.ImageFolder(cfg.data_root / "renders")
    photo_ds = datasets.ImageFolder(cfg.data_root / "photos")

    print("Render classes:", len(render_ds.class_to_idx))
    print("Photo classes:", len(photo_ds.class_to_idx))

    print("Are mappings identical?",
        render_ds.class_to_idx == photo_ds.class_to_idx)

    if render_ds.class_to_idx != photo_ds.class_to_idx:
        raise RuntimeError("Render and photo class mappings are not identical.")

    # Save class-to-index mapping for inference
    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/class_to_idx.json", "w") as f:
        json.dump(render_ds.class_to_idx, f, indent=4)

    print("[INFO] Saved class mapping to artifacts/class_to_idx.json")


    # Baseline evaluation before training
    print("[INFO] Baseline evaluation before training")

    print("[BASELINE] Renders:")
    eval_one_epoch(model, render_val_loader, device, epoch=-1)

    print("[BASELINE] Real Photos:")
    eval_one_epoch(model, val_loader, device, epoch=-1)


    # Stage 1: Pretrain on renders
    print("[INFO] Stage 1: Pretrain on renders")

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4
    )

    for epoch in range(cfg.epochs_renders):

        tr_l, tr_a = train_one_epoch(
            model,
            render_train_loader,
            optimizer,
            device,
            scaler,
            epoch,
            "RENDERS"
        )

        # Validation on renders
        va_l_r, va_a_r = eval_one_epoch(model, render_val_loader, device, epoch)

        # Validation on real photos
        va_l_p, va_a_p = eval_one_epoch(model, val_loader, device, epoch)

        print(f"[H1] {epoch+1}: "
              f"train_acc={tr_a:.4f} "
              f"val_render={va_a_r:.4f} "
              f"val_real={va_a_p:.4f}")

        # Save history
        history["renders"]["train_loss"].append(tr_l)
        history["renders"]["train_acc"].append(tr_a)

        history["renders"]["val_render_loss"].append(va_l_r)
        history["renders"]["val_render_acc"].append(va_a_r)

        history["renders"]["val_real_loss"].append(va_l_p)
        history["renders"]["val_real_acc"].append(va_a_p)


    # Stage 1.5: Train head only
    print("[INFO] Stage 1.5: Adapt head")

    for p in model.parameters():
        p.requires_grad = False

    for p in model.fc.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=1e-3
    )

    for epoch in range(3):

        train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            epoch,
            "HEAD_ADAPT",
            class_weights
        )


    # Stage 2: Fine-tune
    print("[INFO] Stage 2: Fine-tune")

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.fc.parameters(), "lr": cfg.lr_head},
        {"params": model.layer4.parameters(), "lr": cfg.lr_backbone},
    ])

    best_acc = 0

    for epoch in range(cfg.epochs_photos_unfreeze):

        tr_l, tr_a = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            epoch,
            "FINETUNE",
            class_weights
        )

        va_l, va_a = eval_one_epoch(
            model,
            val_loader,
            device,
            epoch
        )

        print(f"[FT] {epoch+1}: "
              f"train_acc={tr_a:.4f} "
              f"val_acc={va_a:.4f}")

        if va_a > best_acc:
            best_acc = va_a
            torch.save(
                model.state_dict(),
                cfg.save_dir / "best_model.pt"
            )

        history["photos"]["train_loss"].append(tr_l)
        history["photos"]["train_acc"].append(tr_a)
        history["photos"]["val_loss"].append(va_l)
        history["photos"]["val_acc"].append(va_a)


    # Save history
    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/training_history.json", "w") as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()
