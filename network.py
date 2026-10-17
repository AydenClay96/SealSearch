"""Identify seal IDs with pretrained visual embeddings and nearest references."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARCHITECTURE = "resnet50-layer3-embedding-v1"


class SealDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load and cache images from ``split/class_name/image`` folders."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        class_names: Sequence[str] | None = None,
        image_size: int = 448,
        excluded_hashes: set[str] | None = None,
    ) -> None:
        self.split_dir = Path(data_dir) / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {self.split_dir}")

        discovered_classes = sorted(
            path.name for path in self.split_dir.iterdir() if path.is_dir()
        )
        if not discovered_classes:
            raise ValueError(f"No class folders found in {self.split_dir}")

        self.class_names = list(class_names or discovered_classes)
        unexpected_classes = set(discovered_classes) - set(self.class_names)
        if unexpected_classes:
            raise ValueError(
                f"Unexpected class folders in {self.split_dir}: "
                f"{', '.join(sorted(unexpected_classes))}"
            )
        self.class_to_index = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }
        self.image_size = image_size
        self.excluded_count = 0
        self.samples: list[tuple[Path, int]] = []
        self.image_hashes: set[str] = set()
        cached_images: list[torch.Tensor] = []
        seen_hashes: dict[str, tuple[Path, int]] = {}
        excluded_hashes = excluded_hashes or set()

        for class_name in self.class_names:
            class_index = self.class_to_index[class_name]
            for image_path in sorted((self.split_dir / class_name).iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image_bytes = image_path.read_bytes()
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                if image_hash in excluded_hashes:
                    self.excluded_count += 1
                    continue
                previous_sample = seen_hashes.get(image_hash)
                if previous_sample is not None:
                    if previous_sample[1] != class_index:
                        raise ValueError(
                            "Identical images have conflicting labels: "
                            f"{previous_sample[0]} and {image_path}"
                        )
                    self.excluded_count += 1
                    continue

                image = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = _resize_with_padding(image, image_size)
                cached_images.append(
                    torch.from_numpy(image).permute(2, 0, 1).contiguous()
                )
                self.samples.append((image_path, class_index))
                self.image_hashes.add(image_hash)
                seen_hashes[image_hash] = (image_path, class_index)

        if not self.samples:
            raise ValueError(f"No supported images found in {self.split_dir}")
        self.images = torch.stack(cached_images)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.samples[index][1]


class SealEmbeddingNetwork(nn.Module):
    """Frozen ImageNet ResNet-50 features for individual-seal matching."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        features = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(images))))
        features = backbone.layer1(features)
        features = backbone.layer2(features)
        features = backbone.layer3(features)
        pooled = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        return F.normalize(pooled, p=2, dim=1)


@dataclass(frozen=True)
class EvaluationMetrics:
    top1_accuracy: float
    top3_accuracy: float
    macro_accuracy: float
    num_images: int


@dataclass(frozen=True)
class TrainingResult:
    model: SealEmbeddingNetwork
    class_names: list[str]
    reference_embeddings: torch.Tensor
    reference_labels: torch.Tensor
    validation_metrics: EvaluationMetrics
    device: torch.device


@dataclass(frozen=True)
class Prediction:
    seal_id: str
    similarity: float


def train_model(
    data_dir: str | Path = "output",
    checkpoint_path: str | Path = "seal_model.pth",
    batch_size: int = 16,
    image_size: int = 448,
    device: str | torch.device | None = None,
) -> TrainingResult:
    """Build a nearest-reference index from the training split."""
    if batch_size < 1 or image_size < 64:
        raise ValueError("batch_size must be positive and image_size must be at least 64")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if selected_device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))

    train_data = SealDataset(data_dir, "train", image_size=image_size)
    val_data = SealDataset(
        data_dir,
        "val",
        class_names=train_data.class_names,
        image_size=image_size,
        excluded_hashes=train_data.image_hashes,
    )
    if val_data.excluded_count:
        print(
            f"Excluded {val_data.excluded_count} exact duplicate image(s) "
            "from validation to prevent train/validation leakage"
        )

    model = SealEmbeddingNetwork(pretrained=True).to(
        selected_device, memory_format=torch.channels_last
    )
    model.requires_grad_(False)
    model.eval()
    device_name = (
        torch.cuda.get_device_name(selected_device)
        if selected_device.type == "cuda"
        else str(selected_device)
    )
    print(
        f"Building {ARCHITECTURE} reference index on {device_name} "
        f"({len(train_data)} images, {len(train_data.class_names)} seal IDs)"
    )

    reference_embeddings = _extract_embeddings(
        model, train_data, batch_size, selected_device
    )
    reference_labels = torch.tensor(
        [label for _, label in train_data.samples], dtype=torch.long
    )
    validation_embeddings = _extract_embeddings(
        model, val_data, batch_size, selected_device
    )
    validation_labels = torch.tensor(
        [label for _, label in val_data.samples], dtype=torch.long
    )
    validation_metrics = _evaluate_embeddings(
        validation_embeddings,
        validation_labels,
        reference_embeddings,
        reference_labels,
        len(train_data.class_names),
        selected_device,
    )

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "class_names": train_data.class_names,
            "image_size": image_size,
            "reference_embeddings": reference_embeddings.half(),
            "reference_labels": reference_labels,
        },
        checkpoint,
    )
    print(
        f"Validation top-1={validation_metrics.top1_accuracy:.1%}, "
        f"top-3={validation_metrics.top3_accuracy:.1%}, "
        f"macro={validation_metrics.macro_accuracy:.1%} "
        f"({validation_metrics.num_images} leakage-free images)"
    )
    return TrainingResult(
        model=model,
        class_names=train_data.class_names,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        validation_metrics=validation_metrics,
        device=selected_device,
    )


def evaluate_model(
    model: SealEmbeddingNetwork,
    data_dir: str | Path,
    class_names: Sequence[str],
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    split: str = "test",
    image_size: int = 448,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> EvaluationMetrics:
    """Evaluate using nearest references, excluding exact images from prior splits."""
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    excluded_hashes: set[str] = set()
    for earlier_split in ("train", "val"):
        if split == earlier_split:
            break
        excluded_hashes.update(_split_image_hashes(Path(data_dir) / earlier_split))

    dataset = SealDataset(
        data_dir,
        split,
        class_names=class_names,
        image_size=image_size,
        excluded_hashes=excluded_hashes,
    )
    if dataset.excluded_count:
        print(
            f"Excluded {dataset.excluded_count} exact duplicate image(s) "
            f"from {split} because they appear in an earlier split"
        )

    query_embeddings = _extract_embeddings(
        model.to(selected_device), dataset, batch_size, selected_device
    )
    query_labels = torch.tensor(
        [label for _, label in dataset.samples], dtype=torch.long
    )
    return _evaluate_embeddings(
        query_embeddings,
        query_labels,
        reference_embeddings,
        reference_labels,
        len(class_names),
        selected_device,
    )


def predict_seal(
    image: str | Path | np.ndarray,
    checkpoint_path: str | Path = "seal_model.pth",
    top_k: int = 3,
    device: str | torch.device | None = None,
) -> list[Prediction]:
    """Return seal IDs ranked by nearest-reference cosine similarity."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(
            "This checkpoint uses an older network architecture; "
            "rebuild it with `python network.py train`"
        )

    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])
    model = SealEmbeddingNetwork().to(
        selected_device, memory_format=torch.channels_last
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if isinstance(image, (str, Path)):
        image_array = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if image_array is None:
            raise ValueError(f"Could not read image: {image}")
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        image_array = np.asarray(image)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("Image arrays must have shape (height, width, 3)")
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    image_array = _resize_with_padding(image_array, image_size)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    with torch.inference_mode():
        query = _prepare_images(image_tensor.unsqueeze(0), selected_device)
        query = query.to(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=selected_device.type,
            dtype=torch.float16,
            enabled=selected_device.type == "cuda",
        ):
            embedding = model(query).float()

    references = checkpoint["reference_embeddings"].to(
        selected_device, dtype=torch.float32
    )
    reference_labels = checkpoint["reference_labels"].to(selected_device)
    class_scores = _class_similarity_scores(
        embedding,
        references,
        reference_labels,
        len(class_names),
    )[0]
    scores, indices = class_scores.topk(min(top_k, len(class_names)))
    return [
        Prediction(class_names[index], float(score))
        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
    ]


def _extract_embeddings(
    model: SealEmbeddingNetwork,
    dataset: SealDataset,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batches = []
    with torch.inference_mode():
        for images, _ in loader:
            images = _prepare_images(images, device).to(
                memory_format=torch.channels_last
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                batches.append(model(images).float())
    return torch.cat(batches).cpu()


def _evaluate_embeddings(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> EvaluationMetrics:
    query_embeddings = query_embeddings.to(device)
    query_labels = query_labels.to(device)
    reference_embeddings = reference_embeddings.to(device)
    reference_labels = reference_labels.to(device)
    class_scores = _class_similarity_scores(
        query_embeddings, reference_embeddings, reference_labels, num_classes
    )
    rankings = class_scores.argsort(dim=1, descending=True)
    correct = rankings[:, 0] == query_labels
    top3_correct = (rankings[:, : min(3, num_classes)] == query_labels[:, None]).any(
        dim=1
    )
    class_counts = torch.bincount(query_labels, minlength=num_classes)
    class_correct = torch.bincount(
        query_labels[correct], minlength=num_classes
    )
    present_classes = class_counts > 0
    macro_accuracy = (
        class_correct[present_classes] / class_counts[present_classes]
    ).mean()
    return EvaluationMetrics(
        top1_accuracy=correct.float().mean().item(),
        top3_accuracy=top3_correct.float().mean().item(),
        macro_accuracy=macro_accuracy.item(),
        num_images=len(query_labels),
    )


def _class_similarity_scores(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    reference_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    similarities = query_embeddings @ reference_embeddings.T
    return torch.stack(
        [
            similarities[:, reference_labels == class_index].max(dim=1).values
            for class_index in range(num_classes)
        ],
        dim=1,
    )


def _split_image_hashes(split_dir: Path) -> set[str]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {split_dir}")
    return {
        hashlib.sha256(image_path.read_bytes()).hexdigest()
        for image_path in split_dir.rglob("*")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True).float().div_(255.0)
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def _resize_with_padding(image: np.ndarray, image_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = image_size / max(height, width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="build a reference index for seal IDs")
    train_parser.add_argument("--data", type=Path, default=Path("output"))
    train_parser.add_argument("--checkpoint", type=Path, default=Path("seal_model.pth"))
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--image-size", type=int, default=448)
    predict_parser = commands.add_parser("predict", help="identify a seal in an image")
    predict_parser.add_argument("image", type=Path)
    predict_parser.add_argument("--checkpoint", type=Path, default=Path("seal_model.pth"))
    predict_parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.command == "train":
        result = train_model(
            args.data,
            args.checkpoint,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        test_metrics = evaluate_model(
            result.model,
            args.data,
            result.class_names,
            result.reference_embeddings,
            result.reference_labels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=result.device,
        )
        print(
            f"Test top-1: {test_metrics.top1_accuracy:.1%}; "
            f"top-3: {test_metrics.top3_accuracy:.1%}; "
            f"macro accuracy: {test_metrics.macro_accuracy:.1%} "
            f"({test_metrics.num_images} leakage-free images)"
        )
    else:
        for prediction in predict_seal(
            args.image, args.checkpoint, top_k=args.top_k
        ):
            print(f"{prediction.seal_id}: cosine similarity {prediction.similarity:.3f}")


if __name__ == "__main__":
    _main()
