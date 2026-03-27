"""Transfer learning con EfficientNet para clasificacion emparejada de artefactos.

Cada muestra esta formada por:
- una foto
- un dibujo
- una clase de forma

La arquitectura comparte un backbone EfficientNet entre ambas modalidades,
fusiona sus embeddings y aprende una cabeza final especializada en el dataset
de artefactos del proyecto.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

try:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    HAS_EFFICIENTNET_WEIGHTS_API = True
except ImportError:
    from torchvision.models import efficientnet_b0

    EfficientNet_B0_Weights = None
    HAS_EFFICIENTNET_WEIGHTS_API = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "cssl_dataset"
BASE_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIR = BASE_OUTPUT_DIR / "checkpoints"
DEFAULT_HISTORY_DIR = BASE_OUTPUT_DIR / "historial"
DEFAULT_VISUAL_DIR = BASE_OUTPUT_DIR / "pruebas visuales"
ITEM_UUID_PATTERN = re.compile(r"itemUUID_([^_]+)")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-").lower()


def load_grayscale_image(image_path: Path) -> np.ndarray:
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if file_bytes.size == 0:
        raise FileNotFoundError(f"No se pudo leer el fichero: {image_path}")
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")
    return image


def save_rgb_image(image_path: Path, image: np.ndarray) -> None:
    ensure_dir(image_path.parent)
    success, encoded = cv2.imencode(image_path.suffix or ".png", image)
    if not success:
        raise OSError(f"No se pudo codificar la imagen para guardar en {image_path}")
    encoded.tofile(str(image_path))


def read_split_file(split_path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        file_name, label, split_name = line.split("*")
        rows.append((file_name, label, split_name))
    if not rows:
        raise ValueError(f"No se encontraron entradas en {split_path}")
    return rows


def build_class_to_idx(labels: Sequence[str]) -> Dict[str, int]:
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


def resolve_split_path(
    dataset_root: Path,
    split_name: str,
    modality: str,
    experiment: str,
) -> Path:
    split_path = (
        dataset_root
        / "experiments"
        / "regular_shape"
        / "shapes"
        / "train_sets"
        / split_name
        / modality
        / experiment
        / "split.txt"
    )
    if not split_path.exists():
        raise FileNotFoundError(f"No existe el split solicitado: {split_path}")
    return split_path


def resolve_image_root(dataset_root: Path, modality: str) -> Path:
    if modality == "photos":
        image_root = dataset_root / "all_image_base" / "1"
    elif modality == "drawings":
        image_root = dataset_root / "all_drawing_base" / "1"
    else:
        raise ValueError("La modalidad debe ser 'photos' o 'drawings'")
    if not image_root.exists():
        raise FileNotFoundError(f"No existe el directorio de imagenes: {image_root}")
    return image_root


def extract_item_uuid(file_name: str) -> str:
    match = ITEM_UUID_PATTERN.search(file_name)
    if match is None:
        raise ValueError(f"No se pudo extraer itemUUID de {file_name}")
    return match.group(1)


def require_paired_modalities(modalities: Sequence[str]) -> Tuple[str, str]:
    normalized = tuple(sorted(set(modalities)))
    if normalized != ("drawings", "photos"):
        raise ValueError(
            "Este modulo trabaja con pares y requiere exactamente ambas modalidades: photos y drawings"
        )
    return ("photos", "drawings")


def build_paired_records(
    dataset_root: Path,
    split_name: str,
    experiment: str,
    modalities: Sequence[str],
) -> Tuple[List[Tuple[str, str, str, str, str]], Dict[str, Path]]:
    require_paired_modalities(modalities)
    split_paths = {
        modality: resolve_split_path(dataset_root, split_name, modality, experiment)
        for modality in ("photos", "drawings")
    }

    by_modality: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
    for modality, split_path in split_paths.items():
        rows = read_split_file(split_path)
        index: Dict[str, Tuple[str, str, str]] = {}
        for file_name, label, split_tag in rows:
            item_uuid = extract_item_uuid(file_name)
            index[item_uuid] = (file_name, label, split_tag)
        by_modality[modality] = index

    common_item_ids = sorted(set(by_modality["photos"]) & set(by_modality["drawings"]))
    if not common_item_ids:
        raise ValueError("No se encontraron items emparejados entre photos y drawings")

    paired_records: List[Tuple[str, str, str, str, str]] = []
    for item_uuid in common_item_ids:
        photo_name, photo_label, photo_split = by_modality["photos"][item_uuid]
        drawing_name, drawing_label, drawing_split = by_modality["drawings"][item_uuid]
        if photo_label != drawing_label:
            raise ValueError(f"Etiqueta distinta entre photo/drawing para itemUUID {item_uuid}")
        if photo_split != drawing_split:
            raise ValueError(f"Split distinto entre photo/drawing para itemUUID {item_uuid}")
        paired_records.append((item_uuid, photo_label, photo_split, photo_name, drawing_name))

    return paired_records, split_paths


def subsample_records(
    records: Sequence[Tuple[str, str, str, str, str]],
    max_samples: Optional[int],
    seed: int,
) -> List[Tuple[str, str, str, str, str]]:
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return list(records)

    rng = random.Random(seed)
    grouped: Dict[str, List[Tuple[str, str, str, str, str]]] = defaultdict(list)
    for record in records:
        grouped[record[1]].append(record)

    base_take = max(1, max_samples // max(1, len(grouped)))
    selected: List[Tuple[str, str, str, str, str]] = []
    leftovers: List[Tuple[str, str, str, str, str]] = []
    for label in sorted(grouped):
        group_records = grouped[label][:]
        rng.shuffle(group_records)
        selected.extend(group_records[:base_take])
        leftovers.extend(group_records[base_take:])

    if len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]
    elif len(selected) < max_samples and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])

    rng.shuffle(selected)
    return selected


class CSSLEfficientNetPairedDataset(Dataset):
    """Dataset emparejado para EfficientNet con normalizacion ImageNet."""

    def __init__(
        self,
        dataset_root: Path,
        records: Sequence[Tuple[str, str, str, str, str]],
        class_to_idx: Dict[str, int],
        image_size: int,
        training: bool,
    ) -> None:
        self.photo_root = resolve_image_root(dataset_root, "photos")
        self.drawing_root = resolve_image_root(dataset_root, "drawings")
        self.records = list(records)
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def _to_rgb_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = TF.to_tensor(image)
        tensor = tensor.repeat(3, 1, 1)
        tensor = TF.resize(
            tensor,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return tensor

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, str, str, str]:
        item_uuid, label, _, photo_name, drawing_name = self.records[index]
        photo_path = self.photo_root / photo_name
        drawing_path = self.drawing_root / drawing_name

        photo_tensor = self._to_rgb_tensor(load_grayscale_image(photo_path))
        drawing_tensor = self._to_rgb_tensor(load_grayscale_image(drawing_path))

        if self.training:
            if random.random() < 0.5:
                photo_tensor = TF.hflip(photo_tensor)
                drawing_tensor = TF.hflip(drawing_tensor)
            angle = random.uniform(-8.0, 8.0)
            photo_tensor = TF.rotate(
                photo_tensor,
                angle,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            drawing_tensor = TF.rotate(
                drawing_tensor,
                angle,
                interpolation=TF.InterpolationMode.BILINEAR,
            )

        photo_tensor = self._normalize(photo_tensor)
        drawing_tensor = self._normalize(drawing_tensor)

        return (
            photo_tensor,
            drawing_tensor,
            self.class_to_idx[label],
            item_uuid,
            str(photo_path),
            str(drawing_path),
        )


def make_paired_dataloaders(
    dataset_root: Path,
    split_name: str,
    experiment: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_train_samples: Optional[int],
    max_test_samples: Optional[int],
    seed: int,
    modalities: Sequence[str],
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[str, int], Dict[str, Path]]:
    paired_records, split_paths = build_paired_records(
        dataset_root=dataset_root,
        split_name=split_name,
        experiment=experiment,
        modalities=modalities,
    )
    class_to_idx = build_class_to_idx([record[1] for record in paired_records])

    train_records = [record for record in paired_records if record[2] == "train"]
    test_records = [record for record in paired_records if record[2] == "test"]

    train_records = subsample_records(train_records, max_train_samples, seed)
    test_records = subsample_records(test_records, max_test_samples, seed + 1)

    train_dataset = CSSLEfficientNetPairedDataset(
        dataset_root=dataset_root,
        records=train_records,
        class_to_idx=class_to_idx,
        image_size=image_size,
        training=True,
    )
    test_dataset = CSSLEfficientNetPairedDataset(
        dataset_root=dataset_root,
        records=test_records,
        class_to_idx=class_to_idx,
        image_size=image_size,
        training=False,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}
    return train_loader, test_loader, class_to_idx, dataset_sizes, split_paths


def resolve_weights(weights_name: str) -> Optional[object]:
    normalized = weights_name.lower()
    if normalized in {"default", "imagenet", "imagenet1k"}:
        if HAS_EFFICIENTNET_WEIGHTS_API and EfficientNet_B0_Weights is not None:
            return EfficientNet_B0_Weights.DEFAULT
        return True
    if normalized in {"none", "random"}:
        return None
    raise ValueError(f"weights no soportado: {weights_name}")


class PairedEfficientNetClassifier(nn.Module):
    """Backbone compartido EfficientNet y cabeza de fusion para el par."""

    def __init__(
        self,
        num_classes: int,
        weights_name: str = "default",
        fusion_dim: int = 256,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        weights = resolve_weights(weights_name)
        try:
            if HAS_EFFICIENTNET_WEIGHTS_API:
                backbone = efficientnet_b0(weights=weights)
            else:
                backbone = efficientnet_b0(pretrained=bool(weights))
        except Exception as exc:
            raise RuntimeError(
                "No se pudieron cargar los pesos preentrenados de EfficientNet. "
                "Si solo quieres una prueba tecnica local, usa --weights none."
            ) from exc

        self.backbone = backbone
        self.feature_dim = int(self.backbone.classifier[1].in_features)
        self.backbone.classifier = nn.Identity()

        fused_dim = self.feature_dim * 4
        self.embedding_head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=max(0.10, dropout * 0.5)),
            nn.Linear(fusion_dim, num_classes),
        )

    def encode_single(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def fuse(self, photo_embedding: torch.Tensor, drawing_embedding: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                photo_embedding,
                drawing_embedding,
                torch.abs(photo_embedding - drawing_embedding),
                photo_embedding * drawing_embedding,
            ],
            dim=1,
        )

    def forward(self, photo: torch.Tensor, drawing: torch.Tensor) -> Dict[str, torch.Tensor]:
        photo_embedding = self.encode_single(photo)
        drawing_embedding = self.encode_single(drawing)
        fused_features = self.fuse(photo_embedding, drawing_embedding)
        embedding = self.embedding_head(fused_features)
        logits = self.classifier(embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "photo_embedding": photo_embedding,
            "drawing_embedding": drawing_embedding,
        }


def set_backbone_trainable(model: PairedEfficientNetClassifier, trainable_blocks: int) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False

    if trainable_blocks <= 0:
        return

    features = list(model.backbone.features.children())
    for block in features[-trainable_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


def build_optimizer(
    model: PairedEfficientNetClassifier,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_params: List[nn.Parameter] = []
    backbone_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups: List[Dict[str, object]] = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def compute_class_prototypes(
    model: PairedEfficientNetClassifier,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    embedding_dim: int,
) -> torch.Tensor:
    model.eval()
    embedding_sums = [torch.zeros(embedding_dim, device=device) for _ in range(num_classes)]
    counts = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for photo, drawing, labels, _, _, _ in dataloader:
            photo = photo.to(device)
            drawing = drawing.to(device)
            labels = labels.to(device)
            embeddings = model(photo, drawing)["embedding"]
            for class_idx in range(num_classes):
                mask = labels == class_idx
                if mask.any():
                    embedding_sums[class_idx] += embeddings[mask].sum(dim=0)
                    counts[class_idx] += int(mask.sum().item())

    global_mean = torch.stack(embedding_sums).sum(dim=0) / max(1, sum(counts))
    prototypes: List[torch.Tensor] = []
    for class_idx in range(num_classes):
        if counts[class_idx] == 0:
            prototypes.append(global_mean.clone())
        else:
            prototypes.append(embedding_sums[class_idx] / counts[class_idx])
    return torch.stack(prototypes, dim=0)


def compute_scores(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    prototypes: Optional[torch.Tensor],
    prototype_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probability_costs = 1.0 - torch.softmax(logits, dim=1)
    if prototypes is None or prototype_weight <= 0.0:
        prototype_distances = torch.zeros_like(probability_costs)
        return probability_costs, prototype_distances

    prototype_distances = ((embeddings.unsqueeze(1) - prototypes.unsqueeze(0)) ** 2).mean(dim=2)
    scores = probability_costs + prototype_weight * prototype_distances
    return scores, prototype_distances


def evaluate(
    model: PairedEfficientNetClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    prototypes: Optional[torch.Tensor],
    prototype_weight: float,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_hybrid_corrects = 0
    total = 0

    with torch.no_grad():
        for photo, drawing, labels, _, _, _ in dataloader:
            photo = photo.to(device)
            drawing = drawing.to(device)
            labels = labels.to(device)
            outputs = model(photo, drawing)
            logits = outputs["logits"]
            embeddings = outputs["embedding"]
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            scores, _ = compute_scores(
                logits=logits,
                embeddings=embeddings,
                prototypes=prototypes,
                prototype_weight=prototype_weight,
            )
            hybrid_preds = scores.argmin(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += int((preds == labels).sum().item())
            running_hybrid_corrects += int((hybrid_preds == labels).sum().item())
            total += batch_size

    return (
        running_loss / max(1, total),
        running_corrects / max(1, total),
        running_hybrid_corrects / max(1, total),
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: PairedEfficientNetClassifier,
    optimizer: torch.optim.Optimizer,
    class_to_idx: Dict[str, int],
    config: Dict[str, object],
    prototypes: Optional[torch.Tensor],
    history: List[Dict[str, float]],
    best_val_hybrid_acc: float,
    completed_epochs: int,
) -> None:
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "config": config,
            "prototypes": None if prototypes is None else prototypes.detach().cpu(),
            "history": history,
            "best_val_hybrid_acc": best_val_hybrid_acc,
            "completed_epochs": completed_epochs,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[PairedEfficientNetClassifier, Dict[str, int], Dict[str, object], Optional[torch.Tensor], List[Dict[str, float]]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    class_to_idx = checkpoint["class_to_idx"]
    config = checkpoint["config"]
    model = PairedEfficientNetClassifier(
        num_classes=len(class_to_idx),
        weights_name="none",
        fusion_dim=int(config.get("fusion_dim", 256)),
        dropout=float(config.get("dropout", 0.30)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    prototypes = checkpoint.get("prototypes")
    history = checkpoint.get("history", [])
    return model, class_to_idx, config, prototypes, history


def load_pair_for_prediction(
    photo_path: Path,
    drawing_path: Path,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def prepare(image_path: Path) -> torch.Tensor:
        image = load_grayscale_image(image_path)
        tensor = TF.to_tensor(image).repeat(3, 1, 1)
        tensor = TF.resize(
            tensor,
            [image_size, image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return tensor

    return prepare(photo_path), prepare(drawing_path)


def denormalize_rgb_tensor(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    array = tensor.detach().cpu()
    array = (array * std.cpu()) + mean.cpu()
    array = array.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (array * 255.0).astype(np.uint8)


def predict_pair_tensors(
    model: PairedEfficientNetClassifier,
    photo_tensor: torch.Tensor,
    drawing_tensor: torch.Tensor,
    class_to_idx: Dict[str, int],
    prototypes: Optional[torch.Tensor],
    device: torch.device,
    prototype_weight: float,
) -> Dict[str, object]:
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    model.eval()

    with torch.no_grad():
        outputs = model(
            photo_tensor.unsqueeze(0).to(device),
            drawing_tensor.unsqueeze(0).to(device),
        )
        logits = outputs["logits"]
        embedding = outputs["embedding"]
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, prototype_distances = compute_scores(
            logits=logits,
            embeddings=embedding,
            prototypes=None if prototypes is None else prototypes.to(device),
            prototype_weight=prototype_weight,
        )

    scores_1d = scores[0]
    proto_1d = prototype_distances[0]
    ordering = torch.argsort(scores_1d).tolist()
    best_idx = int(ordering[0])
    class_scores: List[Dict[str, object]] = []
    for class_idx in ordering:
        class_scores.append(
            {
                "class_idx": class_idx,
                "label": idx_to_class[class_idx],
                "score": float(scores_1d[class_idx].item()),
                "probability": float(probabilities[class_idx].item()),
                "prototype_distance": float(proto_1d[class_idx].item()),
            }
        )

    return {
        "predicted_idx": best_idx,
        "predicted_label": idx_to_class[best_idx],
        "scores": class_scores,
    }


def predict_pair(
    photo_path: Path,
    drawing_path: Path,
    checkpoint_path: Path,
    prototype_weight: Optional[float] = None,
    force_cpu: bool = False,
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    model, class_to_idx, config, prototypes, _ = load_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    photo_tensor, drawing_tensor = load_pair_for_prediction(
        Path(photo_path),
        Path(drawing_path),
        int(config["image_size"]),
    )
    return predict_pair_tensors(
        model=model,
        photo_tensor=photo_tensor,
        drawing_tensor=drawing_tensor,
        class_to_idx=class_to_idx,
        prototypes=prototypes,
        device=device,
        prototype_weight=float(
            prototype_weight if prototype_weight is not None else config.get("prototype_weight", 0.0)
        ),
    )


def select_visual_indices(
    dataset: CSSLEfficientNetPairedDataset,
    num_examples: int,
    seed: int,
) -> List[int]:
    if num_examples >= len(dataset):
        return list(range(len(dataset)))

    rng = random.Random(seed)
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        grouped[record[1]].append(index)

    indices: List[int] = []
    selected_set = set()
    for label in sorted(grouped):
        group_indices = grouped[label][:]
        rng.shuffle(group_indices)
        if group_indices:
            indices.append(group_indices[0])
            selected_set.add(group_indices[0])
        if len(indices) >= num_examples:
            break

    remaining = [idx for idx in range(len(dataset)) if idx not in selected_set]
    rng.shuffle(remaining)
    indices.extend(remaining[: max(0, num_examples - len(indices))])
    return sorted(indices[:num_examples])


def wrap_text_lines(
    text: str,
    max_width: int,
    font: int,
    font_scale: float,
    thickness: int,
) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if candidate_width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def create_summary_mosaic(
    image_paths: Sequence[Path],
    output_path: Path,
    thumb_scale: float = 0.33,
    columns: int = 2,
) -> None:
    if not image_paths:
        return

    loaded_images: List[np.ndarray] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is not None:
            loaded_images.append(image)
    if not loaded_images:
        return

    thumbnails = [
        cv2.resize(image, dsize=None, fx=thumb_scale, fy=thumb_scale, interpolation=cv2.INTER_AREA)
        for image in loaded_images
    ]
    tile_h = max(image.shape[0] for image in thumbnails)
    tile_w = max(image.shape[1] for image in thumbnails)
    rows = (len(thumbnails) + columns - 1) // columns
    padding = 16
    canvas = np.full(
        (
            rows * tile_h + (rows + 1) * padding,
            columns * tile_w + (columns + 1) * padding,
            3,
        ),
        250,
        dtype=np.uint8,
    )

    for index, thumb in enumerate(thumbnails):
        row = index // columns
        col = index % columns
        y = padding + row * (tile_h + padding)
        x = padding + col * (tile_w + padding)
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb

    save_rgb_image(output_path, canvas)


def render_prediction_board(
    photo_tensor: torch.Tensor,
    drawing_tensor: torch.Tensor,
    predicted_label: str,
    true_label: str,
    class_scores: Sequence[Dict[str, object]],
) -> np.ndarray:
    photo = denormalize_rgb_tensor(photo_tensor)
    drawing = denormalize_rgb_tensor(drawing_tensor)

    panel_h, panel_w = photo.shape[:2]
    separator = 20
    width = panel_w * 2 + separator * 3

    font = cv2.FONT_HERSHEY_SIMPLEX
    border_color = (50, 180, 90) if predicted_label == true_label else (40, 70, 220)
    text_specs: List[Tuple[str, float, int, Tuple[int, int, int]]] = [
        (f"Prediccion: {predicted_label}", 0.86, 2, border_color),
        (f"Etiqueta real: {true_label}", 0.72, 2, (40, 40, 40)),
    ]
    for rank, score_info in enumerate(class_scores[:3], start=1):
        score_text = (
            f"Top {rank}: {score_info['label']} | "
            f"score={score_info['score']:.4f} | "
            f"prob={score_info['probability']:.4f} | "
            f"proto={score_info['prototype_distance']:.4f}"
        )
        text_specs.append((score_text, 0.56, 1, (65, 65, 65)))

    wrapped_lines: List[Tuple[str, float, int, Tuple[int, int, int]]] = []
    max_text_width = width - 2 * separator
    for text, scale, thickness, color in text_specs:
        for line in wrap_text_lines(text, max_text_width, font, scale, thickness):
            wrapped_lines.append((line, scale, thickness, color))

    line_height = max(
        cv2.getTextSize("Ag", font, scale, thickness)[0][1] + 10
        for _, scale, thickness, _ in wrapped_lines
    )
    top_margin = 40 + len(wrapped_lines) * line_height + 20
    height = top_margin + panel_h + separator * 2
    board = np.full((height, width, 3), 245, dtype=np.uint8)

    current_y = 36
    for line, scale, thickness, color in wrapped_lines:
        cv2.putText(board, line, (separator, current_y), font, scale, color, thickness, cv2.LINE_AA)
        current_y += line_height

    photo_bgr = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
    drawing_bgr = cv2.cvtColor(drawing, cv2.COLOR_RGB2BGR)

    x_left = separator
    x_right = panel_w + separator * 2
    y_top = top_margin
    board[y_top : y_top + panel_h, x_left : x_left + panel_w] = photo_bgr
    board[y_top : y_top + panel_h, x_right : x_right + panel_w] = drawing_bgr
    cv2.rectangle(board, (x_left, y_top), (x_left + panel_w - 1, y_top + panel_h - 1), (60, 60, 60), 2)
    cv2.rectangle(board, (x_right, y_top), (x_right + panel_w - 1, y_top + panel_h - 1), (60, 60, 60), 2)
    cv2.putText(board, "Foto", (x_left + 10, y_top + 28), font, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(board, "Dibujo", (x_right + 10, y_top + 28), font, 0.72, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.rectangle(board, (0, 0), (board.shape[1] - 1, board.shape[0] - 1), border_color, 5)
    return board


def create_visual_tests(
    checkpoint_path: Path,
    dataset_root: Optional[Path] = None,
    split_name: Optional[str] = None,
    experiment: Optional[str] = None,
    modalities: Optional[Sequence[str]] = None,
    image_size: Optional[int] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    max_test_samples: int = 0,
    prototype_weight: Optional[float] = None,
    visual_dir: Path = DEFAULT_VISUAL_DIR,
    num_examples: int = 8,
    seed: int = 42,
    force_cpu: bool = False,
) -> Path:
    visual_dir = Path(visual_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    model, class_to_idx, config, prototypes, _ = load_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)

    dataset_root = Path(dataset_root or config["dataset_root"])
    split_name = str(split_name or config["split_name"])
    experiment = str(experiment or config["experiment"])
    modalities = require_paired_modalities(modalities or config["modalities"])
    image_size = int(image_size or config["image_size"])
    prototype_weight = float(
        prototype_weight if prototype_weight is not None else config.get("prototype_weight", 0.0)
    )

    _, test_loader, class_to_idx, _, _ = make_paired_dataloaders(
        dataset_root=dataset_root,
        split_name=split_name,
        experiment=experiment,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_samples=0,
        max_test_samples=max_test_samples,
        seed=seed,
        modalities=modalities,
    )

    test_dataset = test_loader.dataset
    if not isinstance(test_dataset, CSSLEfficientNetPairedDataset):
        raise TypeError("Se esperaba un dataset emparejado para las pruebas visuales")

    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    indices = select_visual_indices(test_dataset, num_examples=num_examples, seed=seed)
    run_name = str(config["run_name"])
    output_dir = ensure_dir(visual_dir / run_name)
    image_paths: List[Path] = []
    summary_items: List[Dict[str, object]] = []

    for visual_order, dataset_index in enumerate(indices, start=1):
        photo_tensor, drawing_tensor, label_idx, item_uuid, photo_path, drawing_path = test_dataset[dataset_index]
        prediction = predict_pair_tensors(
            model=model,
            photo_tensor=photo_tensor,
            drawing_tensor=drawing_tensor,
            class_to_idx=class_to_idx,
            prototypes=prototypes,
            device=device,
            prototype_weight=prototype_weight,
        )
        true_label = idx_to_class[label_idx]
        predicted_label = str(prediction["predicted_label"])
        board = render_prediction_board(
            photo_tensor=photo_tensor,
            drawing_tensor=drawing_tensor,
            predicted_label=predicted_label,
            true_label=true_label,
            class_scores=prediction["scores"],
        )
        output_path = output_dir / (
            f"ejemplo_{visual_order:02d}_{slugify(item_uuid)}"
            f"_pred_{slugify(predicted_label)}_real_{slugify(true_label)}.png"
        )
        save_rgb_image(output_path, board)
        image_paths.append(output_path)
        summary_items.append(
            {
                "item_uuid": item_uuid,
                "photo_path": photo_path,
                "drawing_path": drawing_path,
                "predicted_label": predicted_label,
                "true_label": true_label,
                "scores": prediction["scores"],
                "board_path": str(output_path),
            }
        )

    create_summary_mosaic(
        image_paths=image_paths,
        output_path=output_dir / "mosaico_resumen.png",
    )
    (output_dir / "resumen_pruebas.json").write_text(
        json.dumps(summary_items, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


def train_efficientnet(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    split_name: str = "80-20",
    experiment: str = "experiment_0",
    modalities: Sequence[str] = ("photos", "drawings"),
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    epochs: int = 10,
    freeze_epochs: int = 3,
    unfreeze_blocks: int = 2,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    fusion_dim: int = 256,
    dropout: float = 0.30,
    weights_name: str = "default",
    prototype_weight: float = 0.15,
    max_train_samples: int = 0,
    max_test_samples: int = 0,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    history_dir: Path = DEFAULT_HISTORY_DIR,
    visual_dir: Path = DEFAULT_VISUAL_DIR,
    visual_examples: int = 8,
    seed: int = 42,
    force_cpu: bool = False,
) -> Dict[str, object]:
    dataset_root = Path(dataset_root)
    checkpoint_dir = Path(checkpoint_dir)
    history_dir = Path(history_dir)
    visual_dir = Path(visual_dir)
    modalities = require_paired_modalities(modalities)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    train_loader, test_loader, class_to_idx, dataset_sizes, split_paths = make_paired_dataloaders(
        dataset_root=dataset_root,
        split_name=split_name,
        experiment=experiment,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        seed=seed,
        modalities=modalities,
    )

    model = PairedEfficientNetClassifier(
        num_classes=len(class_to_idx),
        weights_name=weights_name,
        fusion_dim=fusion_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    run_name = f"efficientnet_pair_{split_name}_{experiment}"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    history_path = history_dir / f"{run_name}.json"

    print(f"Entrenando en dispositivo: {device}")
    for modality, split_path in split_paths.items():
        print(f"Split {modality}: {split_path}")
    print(f"Clases: {list(class_to_idx.keys())}")
    print(f"Pares usados -> train: {dataset_sizes['train']}, test: {dataset_sizes['test']}")

    history: List[Dict[str, float]] = []
    best_val_hybrid_acc = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_prototypes: Optional[torch.Tensor] = None

    set_backbone_trainable(model, trainable_blocks=0)
    optimizer = build_optimizer(
        model=model,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1 and unfreeze_blocks > 0:
            set_backbone_trainable(model, trainable_blocks=unfreeze_blocks)
            optimizer = build_optimizer(
                model=model,
                head_lr=head_lr,
                backbone_lr=backbone_lr,
                weight_decay=weight_decay,
            )

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for photo, drawing, labels, _, _, _ in train_loader:
            photo = photo.to(device)
            drawing = drawing.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(photo, drawing)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            batch_size_current = labels.size(0)
            running_loss += loss.item() * batch_size_current
            running_corrects += int((preds == labels).sum().item())
            total += batch_size_current

        train_loss = running_loss / max(1, total)
        train_acc = running_corrects / max(1, total)
        prototypes = compute_class_prototypes(
            model=model,
            dataloader=train_loader,
            device=device,
            num_classes=len(class_to_idx),
            embedding_dim=fusion_dim,
        )
        val_loss, val_acc, val_hybrid_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            prototypes=prototypes,
            prototype_weight=prototype_weight,
        )

        phase_name = "head_only" if epoch <= freeze_epochs else "fine_tuning"
        history_entry = {
            "epoch": epoch,
            "phase": 0.0 if phase_name == "head_only" else 1.0,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_hybrid_acc": val_hybrid_acc,
        }
        history.append(history_entry)

        print(
            f"Epoch {epoch:02d}/{epochs} | fase={phase_name} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_hybrid_acc={val_hybrid_acc:.4f}"
        )

        if val_hybrid_acc > best_val_hybrid_acc:
            best_val_hybrid_acc = val_hybrid_acc
            best_prototypes = prototypes.detach().cpu()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    config = {
        "dataset_root": str(dataset_root),
        "split_name": split_name,
        "experiment": experiment,
        "modalities": list(modalities),
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "freeze_epochs": freeze_epochs,
        "unfreeze_blocks": unfreeze_blocks,
        "head_lr": head_lr,
        "backbone_lr": backbone_lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "fusion_dim": fusion_dim,
        "dropout": dropout,
        "weights_name": weights_name,
        "prototype_weight": prototype_weight,
        "num_workers": num_workers,
        "max_train_samples": max_train_samples,
        "max_test_samples": max_test_samples,
        "seed": seed,
        "run_name": run_name,
    }

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        class_to_idx=class_to_idx,
        config=config,
        prototypes=best_prototypes,
        history=history,
        best_val_hybrid_acc=best_val_hybrid_acc,
        completed_epochs=epochs,
    )
    ensure_dir(history_path.parent)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    visual_output_dir: Optional[Path] = None
    if visual_examples > 0 and dataset_sizes["test"] > 0:
        visual_output_dir = create_visual_tests(
            checkpoint_path=checkpoint_path,
            visual_dir=visual_dir,
            num_examples=min(visual_examples, dataset_sizes["test"]),
            force_cpu=force_cpu,
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "visual_output_dir": None if visual_output_dir is None else str(visual_output_dir),
        "best_val_hybrid_acc": best_val_hybrid_acc,
        "class_to_idx": class_to_idx,
        "dataset_sizes": dataset_sizes,
        "config": config,
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer learning con EfficientNet sobre pares foto-dibujo de artefactos."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena el clasificador EfficientNet emparejado")
    train_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    train_parser.add_argument("--split-name", default="80-20")
    train_parser.add_argument("--experiment", default="experiment_0")
    train_parser.add_argument(
        "--modalities",
        nargs="+",
        default=["photos", "drawings"],
        choices=["photos", "drawings"],
    )
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--freeze-epochs", type=int, default=3)
    train_parser.add_argument("--unfreeze-blocks", type=int, default=2)
    train_parser.add_argument("--head-lr", type=float, default=1e-3)
    train_parser.add_argument("--backbone-lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--label-smoothing", type=float, default=0.05)
    train_parser.add_argument("--fusion-dim", type=int, default=256)
    train_parser.add_argument("--dropout", type=float, default=0.30)
    train_parser.add_argument("--weights", default="default", choices=["default", "imagenet", "none"])
    train_parser.add_argument("--prototype-weight", type=float, default=0.15)
    train_parser.add_argument("--max-train-samples", type=int, default=0)
    train_parser.add_argument("--max-test-samples", type=int, default=0)
    train_parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    train_parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    train_parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL_DIR)
    train_parser.add_argument("--visual-examples", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--force-cpu", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Predice la clase de un par foto-dibujo")
    predict_parser.add_argument("photo_path", type=Path)
    predict_parser.add_argument("drawing_path", type=Path)
    predict_parser.add_argument("checkpoint_path", type=Path)
    predict_parser.add_argument("--prototype-weight", type=float, default=None)
    predict_parser.add_argument("--force-cpu", action="store_true")

    visual_parser = subparsers.add_parser("visual-tests", help="Genera ejemplos visuales del split de test")
    visual_parser.add_argument("checkpoint_path", type=Path)
    visual_parser.add_argument("--dataset-root", type=Path, default=None)
    visual_parser.add_argument("--split-name", default=None)
    visual_parser.add_argument("--experiment", default=None)
    visual_parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        choices=["photos", "drawings"],
    )
    visual_parser.add_argument("--image-size", type=int, default=None)
    visual_parser.add_argument("--batch-size", type=int, default=8)
    visual_parser.add_argument("--num-workers", type=int, default=0)
    visual_parser.add_argument("--max-test-samples", type=int, default=0)
    visual_parser.add_argument("--prototype-weight", type=float, default=None)
    visual_parser.add_argument("--visual-dir", type=Path, default=DEFAULT_VISUAL_DIR)
    visual_parser.add_argument("--num-examples", type=int, default=8)
    visual_parser.add_argument("--seed", type=int, default=42)
    visual_parser.add_argument("--force-cpu", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        result = train_efficientnet(
            dataset_root=args.dataset_root,
            split_name=args.split_name,
            experiment=args.experiment,
            modalities=args.modalities,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            freeze_epochs=args.freeze_epochs,
            unfreeze_blocks=args.unfreeze_blocks,
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            fusion_dim=args.fusion_dim,
            dropout=args.dropout,
            weights_name=args.weights,
            prototype_weight=args.prototype_weight,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            checkpoint_dir=args.checkpoint_dir,
            history_dir=args.history_dir,
            visual_dir=args.visual_dir,
            visual_examples=args.visual_examples,
            seed=args.seed,
            force_cpu=args.force_cpu,
        )
        print(f"Checkpoint guardado en: {result['checkpoint_path']}")
        print(f"Historial guardado en: {result['history_path']}")
        if result["visual_output_dir"] is not None:
            print(f"Pruebas visuales en: {result['visual_output_dir']}")
        return

    if args.command == "predict":
        prediction = predict_pair(
            photo_path=args.photo_path,
            drawing_path=args.drawing_path,
            checkpoint_path=args.checkpoint_path,
            prototype_weight=args.prototype_weight,
            force_cpu=args.force_cpu,
        )
        print(json.dumps(prediction["scores"][:5], indent=2, ensure_ascii=False))
        print(f"Clase predicha: {prediction['predicted_label']}")
        return

    if args.command == "visual-tests":
        output_dir = create_visual_tests(
            checkpoint_path=args.checkpoint_path,
            dataset_root=args.dataset_root,
            split_name=args.split_name,
            experiment=args.experiment,
            modalities=args.modalities,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_test_samples=args.max_test_samples,
            prototype_weight=args.prototype_weight,
            visual_dir=args.visual_dir,
            num_examples=args.num_examples,
            seed=args.seed,
            force_cpu=args.force_cpu,
        )
        print(f"Pruebas visuales generadas en: {output_dir}")
        return

    raise ValueError(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()
