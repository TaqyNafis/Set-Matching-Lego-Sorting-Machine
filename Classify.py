import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_mapping(class_map_path: Path):
    with open(class_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def get_preprocess(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(ckpt_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = build_model(num_classes)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    image_path: Path,
    model: nn.Module,
    preprocess,
    idx_to_class: dict,
    device: torch.device,
    topk: int = 3,
):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)

    top_probs, top_idxs = probs.topk(topk, dim=1)

    results = []
    for p, i in zip(top_probs[0], top_idxs[0]):
        cls_name = idx_to_class[int(i)]
        results.append((cls_name, float(p)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to an input image")
    parser.add_argument("--topk", type=int, default=3, help="Top-K predictions to show")
    parser.add_argument("--img_size", type=int, default=224, help="Resize image to this size")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt",
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--class_map", type=str, default="artifacts/class_to_idx.json",
                        help="Path to class_to_idx.json")
    args = parser.parse_args()

    image_path = Path(args.image)
    ckpt_path = Path(args.ckpt)
    class_map_path = Path(args.class_map)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"class_to_idx.json not found: {class_map_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    _, idx_to_class = load_mapping(class_map_path)
    model = load_model(ckpt_path, num_classes=len(idx_to_class), device=device)
    preprocess = get_preprocess(args.img_size)

    results = predict_image(
        image_path=image_path,
        model=model,
        preprocess=preprocess,
        idx_to_class=idx_to_class,
        device=device,
        topk=args.topk,
    )

    print("\nTop predictions:")
    for cls, conf in results:
        print(f"{cls} -> {conf:.4f}")


if __name__ == "__main__":
    main()
