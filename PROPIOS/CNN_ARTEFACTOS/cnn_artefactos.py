"""Modelos ligeros para clasificacion de artefactos arqueologicos.

Incluye tres arquitecturas sencillas sobre `cssl_dataset`:
- CNN
- MLP
- KAN ligera (implementacion propia basada en bases RBF)

Tambien puede generar un GIF con las curvas de train/val loss de los modelos.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "cssl_dataset"
BASE_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_OUTPUT_DIR / "checkpoints"
DEFAULT_HISTORY_DIR = BASE_OUTPUT_DIR / "historial"
DEFAULT_GIF_DIR = BASE_OUTPUT_DIR / "gifs"

MODEL_DISPLAY_NAMES = {"cnn": "CNN", "mlp": "MLP", "kan": "KAN"}
MODEL_COLORS = {"cnn": "#1f77b4", "mlp": "#d62728", "kan": "#2ca02c"}


def load_grayscale_image(image_path: Path) -> np.ndarray:
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if file_bytes.size == 0:
        raise FileNotFoundError(f"No se pudo leer el fichero: {image_path}")
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")
    return image


class CSSLShapeDataset(Dataset):
    """Dataset ligero para los split de clasificacion por forma."""

    def __init__(
        self,
        image_root: Path,
        records: Sequence[Tuple[str, str]],
        class_to_idx: Dict[str, int],
        image_size: int,
        training: bool,
    ) -> None:
        self.image_root = image_root
        self.records = list(records)
        self.class_to_idx = class_to_idx
        self.training = training

        if training:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=8),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((image_size, image_size)),
                    transforms.Normalize(mean=[0.5], std=[0.5]),
                ]
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        file_name, label = self.records[index]
        image_path = self.image_root / file_name
        image = load_grayscale_image(image_path)
        image_tensor = self.transform(image)
        return image_tensor, self.class_to_idx[label], str(image_path)


class SmallArtefactCNN(nn.Module):
    """CNN compacta para hardware modesto."""

    def __init__(self, num_classes: int, image_size: int) -> None:
        super().__init__()
        del image_size
        self.features = nn.Sequential(
            self._block(1, 16),
            nn.MaxPool2d(kernel_size=2),
            self._block(16, 32),
            nn.MaxPool2d(kernel_size=2),
            self._block(32, 64),
            nn.MaxPool2d(kernel_size=2),
            self._block(64, 96),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class Latent32VAE(nn.Module):
    """VAE ligero que codifica la imagen a un mapa latente de 32x32."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Conv2d(48, 1, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(48, 1, kernel_size=3, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(12, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded).clamp(min=-6.0, max=6.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return {
            "latent": z,
            "latent_mean": mu,
            "latent_logvar": logvar,
            "reconstruction": reconstruction,
        }


class SmallArtefactMLP(nn.Module):
    """MLP ligera sobre un espacio latente 32x32 obtenido con VAE."""

    def __init__(self, num_classes: int, image_size: int) -> None:
        super().__init__()
        del image_size
        self.vae = Latent32VAE()
        in_features = 32 * 32
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        vae_outputs = self.vae(x)
        logits = self.classifier(vae_outputs["latent"])
        return {
            "logits": logits,
            "reconstruction": vae_outputs["reconstruction"],
            "latent_mean": vae_outputs["latent_mean"],
            "latent_logvar": vae_outputs["latent_logvar"],
        }


class RBFKANLayer(nn.Module):
    """Capa KAN ligera basada en expansion RBF aprendible."""

    def __init__(self, in_features: int, out_features: int, num_knots: int = 6) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, num_knots))
        self.log_scale = nn.Parameter(torch.zeros(in_features))
        self.spline_weight = nn.Parameter(
            torch.randn(in_features, num_knots, out_features) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_term = self.base(x)
        scale = torch.exp(self.log_scale).view(1, -1, 1)
        basis = torch.exp(-((x.unsqueeze(-1) - self.grid.view(1, 1, -1)) * scale) ** 2)
        spline_term = torch.einsum("bik,iko->bo", basis, self.spline_weight)
        return base_term + spline_term


class SmallArtefactKAN(nn.Module):
    """KAN compacta sobre un espacio latente 32x32 obtenido con VAE."""

    def __init__(self, num_classes: int, image_size: int) -> None:
        super().__init__()
        del image_size
        self.vae = Latent32VAE()
        self.flatten = nn.Flatten()
        self.kan1 = RBFKANLayer(32 * 32, 96, num_knots=4)
        self.kan2 = RBFKANLayer(96, 48, num_knots=4)
        self.dropout = nn.Dropout(p=0.15)
        self.output = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        vae_outputs = self.vae(x)
        x = self.flatten(vae_outputs["latent"])
        x = F.silu(self.kan1(x))
        x = self.dropout(x)
        x = F.silu(self.kan2(x))
        return {
            "logits": self.output(x),
            "reconstruction": vae_outputs["reconstruction"],
            "latent_mean": vae_outputs["latent_mean"],
            "latent_logvar": vae_outputs["latent_logvar"],
        }


def build_model(model_type: str, num_classes: int, image_size: int) -> nn.Module:
    if model_type == "cnn":
        return SmallArtefactCNN(num_classes=num_classes, image_size=image_size)
    if model_type == "mlp":
        return SmallArtefactMLP(num_classes=num_classes, image_size=image_size)
    if model_type == "kan":
        return SmallArtefactKAN(num_classes=num_classes, image_size=image_size)
    raise ValueError(f"Modelo no soportado: {model_type}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_class_to_idx(rows: Sequence[Tuple[str, str, str]]) -> Dict[str, int]:
    labels = sorted({label for _, label, _ in rows})
    return {label: idx for idx, label in enumerate(labels)}


def subsample_records(
    records: Sequence[Tuple[str, str]],
    max_samples: Optional[int],
    seed: int,
) -> List[Tuple[str, str]]:
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return list(records)

    rng = random.Random(seed)
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record[1]].append(record)

    num_classes = len(grouped)
    base_take = max(1, max_samples // max(1, num_classes))
    selected: List[Tuple[str, str]] = []
    leftovers: List[Tuple[str, str]] = []

    for label in sorted(grouped):
        label_records = grouped[label][:]
        rng.shuffle(label_records)
        selected.extend(label_records[:base_take])
        leftovers.extend(label_records[base_take:])

    if len(selected) > max_samples:
        rng.shuffle(selected)
        selected = selected[:max_samples]
    elif len(selected) < max_samples and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])

    rng.shuffle(selected)
    return selected


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


def make_dataloaders(
    dataset_root: Path,
    split_name: str,
    modality: str,
    experiment: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_train_samples: Optional[int],
    max_test_samples: Optional[int],
    seed: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[str, int], Path]:
    split_path = resolve_split_path(dataset_root, split_name, modality, experiment)
    rows = read_split_file(split_path)
    class_to_idx = build_class_to_idx(rows)
    image_root = resolve_image_root(dataset_root, modality)

    train_records = [(file_name, label) for file_name, label, split in rows if split == "train"]
    test_records = [(file_name, label) for file_name, label, split in rows if split == "test"]

    train_records = subsample_records(train_records, max_train_samples, seed)
    test_records = subsample_records(test_records, max_test_samples, seed + 1)

    train_dataset = CSSLShapeDataset(
        image_root=image_root,
        records=train_records,
        class_to_idx=class_to_idx,
        image_size=image_size,
        training=True,
    )
    test_dataset = CSSLShapeDataset(
        image_root=image_root,
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
    return train_loader, test_loader, class_to_idx, dataset_sizes, split_path


def unpack_model_outputs(model_outputs: object) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(model_outputs, dict):
        logits = model_outputs["logits"]
        aux_outputs = {
            key: value
            for key, value in model_outputs.items()
            if key != "logits" and isinstance(value, torch.Tensor)
        }
        return logits, aux_outputs
    if isinstance(model_outputs, torch.Tensor):
        return model_outputs, {}
    raise TypeError(f"Salida de modelo no soportada: {type(model_outputs)}")


def compute_loss_components(
    logits: torch.Tensor,
    labels: torch.Tensor,
    inputs: torch.Tensor,
    aux_outputs: Dict[str, torch.Tensor],
    criterion: nn.Module,
    vae_recon_weight: float,
    vae_kl_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    classification_loss = criterion(logits, labels)
    reconstruction_loss = torch.tensor(0.0, device=logits.device)
    kl_loss = torch.tensor(0.0, device=logits.device)

    if "reconstruction" in aux_outputs:
        reconstruction = aux_outputs["reconstruction"]
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        mu = aux_outputs["latent_mean"]
        logvar = aux_outputs["latent_logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = (
        classification_loss
        + vae_recon_weight * reconstruction_loss
        + vae_kl_weight * kl_loss
    )
    return total_loss, {
        "classification_loss": float(classification_loss.detach().item()),
        "reconstruction_loss": float(reconstruction_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vae_recon_weight: float,
    vae_kl_weight: float,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    running_classification_loss = 0.0
    running_reconstruction_loss = 0.0
    running_kl_loss = 0.0

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            model_outputs = model(inputs)
            logits, aux_outputs = unpack_model_outputs(model_outputs)
            loss, loss_parts = compute_loss_components(
                logits=logits,
                labels=labels,
                inputs=inputs,
                aux_outputs=aux_outputs,
                criterion=criterion,
                vae_recon_weight=vae_recon_weight,
                vae_kl_weight=vae_kl_weight,
            )
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_classification_loss += loss_parts["classification_loss"] * batch_size
            running_reconstruction_loss += loss_parts["reconstruction_loss"] * batch_size
            running_kl_loss += loss_parts["kl_loss"] * batch_size
            running_corrects += (preds == labels).sum().item()
            total += batch_size

    loss_value = running_loss / max(1, total)
    accuracy = running_corrects / max(1, total)
    metrics = {
        "classification_loss": running_classification_loss / max(1, total),
        "reconstruction_loss": running_reconstruction_loss / max(1, total),
        "kl_loss": running_kl_loss / max(1, total),
    }
    return loss_value, accuracy, metrics


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    config: Dict[str, object],
    history: List[Dict[str, float]],
    best_val_acc: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "config": config,
            "history": history,
            "best_val_acc": best_val_acc,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    model_type = str(config.get("model_type", "cnn"))
    class_to_idx = checkpoint["class_to_idx"]
    model = build_model(
        model_type=model_type,
        num_classes=len(class_to_idx),
        image_size=int(config["image_size"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def save_history(history_path: Path, history: List[Dict[str, float]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def default_checkpoint_path(model_type: str, modality: str, split_name: str, experiment: str) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{model_type}_{modality}_{split_name}_{experiment}.pt"


def default_history_path(model_type: str, modality: str, split_name: str, experiment: str) -> Path:
    return DEFAULT_HISTORY_DIR / f"{model_type}_{modality}_{split_name}_{experiment}.json"


def default_gif_path(modality: str, split_name: str, experiment: str) -> Path:
    return DEFAULT_GIF_DIR / f"comparacion_{modality}_{split_name}_{experiment}.gif"


def default_accuracy_gif_path(modality: str, split_name: str, experiment: str) -> Path:
    return DEFAULT_GIF_DIR / f"comparacion_accuracy_{modality}_{split_name}_{experiment}.gif"


def train_model(
    model_type: str = "cnn",
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    split_name: str = "50-50-q",
    modality: str = "photos",
    experiment: str = "experiment_0",
    image_size: int = 256,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    max_train_samples: Optional[int] = 400,
    max_test_samples: Optional[int] = 200,
    seed: int = 42,
    checkpoint_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
    vae_recon_weight: float = 0.35,
    vae_kl_weight: float = 0.02,
    force_cpu: bool = False,
) -> Dict[str, object]:
    set_seed(seed)
    dataset_root = Path(dataset_root)
    checkpoint_path = checkpoint_path or default_checkpoint_path(
        model_type, modality, split_name, experiment
    )
    history_path = history_path or default_history_path(
        model_type, modality, split_name, experiment
    )
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

    train_loader, test_loader, class_to_idx, dataset_sizes, split_path = make_dataloaders(
        dataset_root=dataset_root,
        split_name=split_name,
        modality=modality,
        experiment=experiment,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        seed=seed,
    )

    model = build_model(model_type=model_type, num_classes=len(class_to_idx), image_size=image_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    best_state = None
    best_val_acc = -1.0

    print(f"Modelo: {MODEL_DISPLAY_NAMES[model_type]}")
    print(f"Entrenando en dispositivo: {device}")
    print(f"Split: {split_path}")
    print(f"Clases: {list(class_to_idx.keys())}")
    print(
        f"Muestras usadas -> train: {dataset_sizes['train']}, "
        f"test: {dataset_sizes['test']}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        running_classification_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kl_loss = 0.0

        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            model_outputs = model(inputs)
            logits, aux_outputs = unpack_model_outputs(model_outputs)
            loss, loss_parts = compute_loss_components(
                logits=logits,
                labels=labels,
                inputs=inputs,
                aux_outputs=aux_outputs,
                criterion=criterion,
                vae_recon_weight=vae_recon_weight,
                vae_kl_weight=vae_kl_weight,
            )
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            batch_size_now = labels.size(0)
            running_loss += loss.item() * batch_size_now
            running_classification_loss += loss_parts["classification_loss"] * batch_size_now
            running_reconstruction_loss += loss_parts["reconstruction_loss"] * batch_size_now
            running_kl_loss += loss_parts["kl_loss"] * batch_size_now
            running_corrects += (preds == labels).sum().item()
            total += batch_size_now

        train_loss = running_loss / max(1, total)
        train_acc = running_corrects / max(1, total)
        train_classification_loss = running_classification_loss / max(1, total)
        train_reconstruction_loss = running_reconstruction_loss / max(1, total)
        train_kl_loss = running_kl_loss / max(1, total)
        val_loss, val_acc, val_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            vae_recon_weight=vae_recon_weight,
            vae_kl_weight=vae_kl_weight,
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_classification_loss": train_classification_loss,
            "train_reconstruction_loss": train_reconstruction_loss,
            "train_kl_loss": train_kl_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_classification_loss": val_metrics["classification_loss"],
            "val_reconstruction_loss": val_metrics["reconstruction_loss"],
            "val_kl_loss": val_metrics["kl_loss"],
        }
        history.append(history_entry)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )
        if model_type in {"mlp", "kan"}:
            print(
                f"  VAE -> train_recon={train_reconstruction_loss:.4f} | "
                f"train_kl={train_kl_loss:.4f} | "
                f"val_recon={val_metrics['reconstruction_loss']:.4f} | "
                f"val_kl={val_metrics['kl_loss']:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    config = {
        "model_type": model_type,
        "dataset_root": str(dataset_root),
        "split_name": split_name,
        "modality": modality,
        "experiment": experiment,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "max_train_samples": max_train_samples,
        "max_test_samples": max_test_samples,
        "seed": seed,
        "vae_recon_weight": vae_recon_weight,
        "vae_kl_weight": vae_kl_weight,
    }

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        class_to_idx=class_to_idx,
        config=config,
        history=history,
        best_val_acc=best_val_acc,
    )
    save_history(history_path=history_path, history=history)

    print(f"Checkpoint guardado en: {checkpoint_path}")
    print(f"Historial guardado en: {history_path}")

    return {
        "model_type": model_type,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "best_val_acc": best_val_acc,
        "class_to_idx": class_to_idx,
        "dataset_sizes": dataset_sizes,
        "config": config,
        "history": history,
    }


def train_cnn(**kwargs: object) -> Dict[str, object]:
    return train_model(model_type="cnn", **kwargs)


def train_mlp(**kwargs: object) -> Dict[str, object]:
    return train_model(model_type="mlp", **kwargs)


def train_kan(**kwargs: object) -> Dict[str, object]:
    return train_model(model_type="kan", **kwargs)


def create_metric_gif(
    histories_by_model: Dict[str, List[Dict[str, float]]],
    gif_path: Path,
    train_metric_key: str,
    val_metric_key: str,
    y_label: str,
    title: str,
    duration: float = 0.8,
) -> Path:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gif_path.parent.mkdir(parents=True, exist_ok=True)

    max_epochs = max(len(history) for history in histories_by_model.values())
    max_metric = max(
        entry[train_metric_key]
        for history in histories_by_model.values()
        for entry in history
    )
    max_metric = max(
        max_metric,
        max(
            entry[val_metric_key] for history in histories_by_model.values() for entry in history
        ),
    )
    frames: List[np.ndarray] = []

    for epoch in range(1, max_epochs + 1):
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_type, history in histories_by_model.items():
            current_history = history[: min(epoch, len(history))]
            epochs_axis = [entry["epoch"] for entry in current_history]
            train_values = [entry[train_metric_key] for entry in current_history]
            val_values = [entry[val_metric_key] for entry in current_history]
            color = MODEL_COLORS[model_type]
            label = MODEL_DISPLAY_NAMES[model_type]

            ax.plot(
                epochs_axis,
                train_values,
                color=color,
                linewidth=2.5,
                label=f"{label} train",
            )
            ax.plot(
                epochs_axis,
                val_values,
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.75,
                label=f"{label} val",
            )

        ax.set_xlim(1, max_epochs)
        ax.set_ylim(0, max_metric * 1.1 if max_metric > 0 else 1.0)
        ax.set_xlabel("Epoca")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} - epoca {epoch}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()

        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape((height, width, 4))[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(gif_path, frames, duration=duration)
    return gif_path


def create_loss_gif(
    histories_by_model: Dict[str, List[Dict[str, float]]],
    gif_path: Path,
    title: str = "Comparacion de losses",
    duration: float = 0.8,
) -> Path:
    return create_metric_gif(
        histories_by_model=histories_by_model,
        gif_path=gif_path,
        train_metric_key="train_loss",
        val_metric_key="val_loss",
        y_label="Loss",
        title=title,
        duration=duration,
    )


def create_accuracy_gif(
    histories_by_model: Dict[str, List[Dict[str, float]]],
    gif_path: Path,
    title: str = "Comparacion de accuracies",
    duration: float = 0.8,
) -> Path:
    return create_metric_gif(
        histories_by_model=histories_by_model,
        gif_path=gif_path,
        train_metric_key="train_acc",
        val_metric_key="val_acc",
        y_label="Accuracy",
        title=title,
        duration=duration,
    )


def load_history(history_path: Path) -> List[Dict[str, float]]:
    history_path = Path(history_path)
    return json.loads(history_path.read_text(encoding="utf-8"))


def histories_from_files(history_paths: Sequence[Path]) -> Dict[str, List[Dict[str, float]]]:
    histories_by_model: Dict[str, List[Dict[str, float]]] = {}
    for history_path in history_paths:
        history_path = Path(history_path)
        model_key = history_path.stem.split("_", 1)[0].lower()
        if model_key not in MODEL_DISPLAY_NAMES:
            raise ValueError(f"No se pudo inferir el modelo desde {history_path.name}")
        histories_by_model[model_key] = load_history(history_path)
    return histories_by_model


def create_gifs_from_history_files(
    history_paths: Sequence[Path],
    loss_gif_path: Path,
    accuracy_gif_path: Path,
    duration: float = 0.8,
) -> Dict[str, str]:
    histories_by_model = histories_from_files(history_paths)
    create_loss_gif(
        histories_by_model=histories_by_model,
        gif_path=loss_gif_path,
        title="Curvas de aprendizaje CNN / MLP / KAN",
        duration=duration,
    )
    create_accuracy_gif(
        histories_by_model=histories_by_model,
        gif_path=accuracy_gif_path,
        title="Curvas de accuracy CNN / MLP / KAN",
        duration=duration,
    )
    return {
        "loss_gif_path": str(loss_gif_path),
        "accuracy_gif_path": str(accuracy_gif_path),
    }


def train_all_models(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    split_name: str = "50-50-q",
    modality: str = "photos",
    experiment: str = "experiment_0",
    image_size: int = 256,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    max_train_samples: Optional[int] = 400,
    max_test_samples: Optional[int] = 200,
    seed: int = 42,
    vae_recon_weight: float = 0.35,
    vae_kl_weight: float = 0.02,
    force_cpu: bool = False,
    gif_path: Optional[Path] = None,
    accuracy_gif_path: Optional[Path] = None,
) -> Dict[str, object]:
    results: Dict[str, Dict[str, object]] = {}
    histories_by_model: Dict[str, List[Dict[str, float]]] = {}

    for index, model_type in enumerate(["cnn", "mlp", "kan"]):
        print(f"\n===== Entrenando {MODEL_DISPLAY_NAMES[model_type]} =====")
        result = train_model(
            model_type=model_type,
            dataset_root=dataset_root,
            split_name=split_name,
            modality=modality,
            experiment=experiment,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_workers=num_workers,
            max_train_samples=max_train_samples,
            max_test_samples=max_test_samples,
            seed=seed + index,
            vae_recon_weight=vae_recon_weight,
            vae_kl_weight=vae_kl_weight,
            force_cpu=force_cpu,
        )
        results[model_type] = result
        histories_by_model[model_type] = result["history"]  # type: ignore[index]

    gif_path = gif_path or default_gif_path(modality, split_name, experiment)
    accuracy_gif_path = accuracy_gif_path or default_accuracy_gif_path(
        modality, split_name, experiment
    )
    create_loss_gif(
        histories_by_model=histories_by_model,
        gif_path=gif_path,
        title="Curvas de aprendizaje CNN / MLP / KAN",
    )
    create_accuracy_gif(
        histories_by_model=histories_by_model,
        gif_path=accuracy_gif_path,
        title="Curvas de accuracy CNN / MLP / KAN",
    )
    print(f"GIF guardado en: {gif_path}")
    print(f"GIF de accuracy guardado en: {accuracy_gif_path}")

    return {
        "results": results,
        "gif_path": str(gif_path),
        "accuracy_gif_path": str(accuracy_gif_path),
    }


def _build_predict_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def predict_image(
    image_path: Path,
    checkpoint_path: Path,
    top_k: int = 3,
    force_cpu: bool = False,
) -> List[Dict[str, float]]:
    """Carga un checkpoint ya entrenado y predice sobre una imagen."""

    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)

    image_size = int(checkpoint["config"]["image_size"])
    idx_to_class = {
        int(idx): label for idx, label in checkpoint["idx_to_class"].items()
    }
    transform = _build_predict_transform(image_size)

    image = load_grayscale_image(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        model_outputs = model(input_tensor)
        logits, _ = unpack_model_outputs(model_outputs)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = min(top_k, probabilities.numel())
    scores, indices = torch.topk(probabilities, k=top_k)

    predictions: List[Dict[str, float]] = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        predictions.append(
            {
                "label": idx_to_class[index],
                "probability": float(score),
            }
        )
    return predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrenamiento y prediccion con CNN, MLP y KAN ligeras sobre cssl_dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena un modelo individual.")
    train_parser.add_argument("--model-type", type=str, choices=["cnn", "mlp", "kan"], default="cnn")
    train_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    train_parser.add_argument("--split-name", type=str, default="50-50-q")
    train_parser.add_argument("--modality", type=str, choices=["photos", "drawings"], default="photos")
    train_parser.add_argument("--experiment", type=str, default="experiment_0")
    train_parser.add_argument("--image-size", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--max-train-samples", type=int, default=400)
    train_parser.add_argument("--max-test-samples", type=int, default=200)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--checkpoint-path", type=Path, default=None)
    train_parser.add_argument("--history-path", type=Path, default=None)
    train_parser.add_argument("--vae-recon-weight", type=float, default=0.35)
    train_parser.add_argument("--vae-kl-weight", type=float, default=0.02)
    train_parser.add_argument("--force-cpu", action="store_true")

    train_all_parser = subparsers.add_parser(
        "train-all", help="Entrena CNN, MLP y KAN y genera un GIF comparativo."
    )
    train_all_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    train_all_parser.add_argument("--split-name", type=str, default="50-50-q")
    train_all_parser.add_argument("--modality", type=str, choices=["photos", "drawings"], default="photos")
    train_all_parser.add_argument("--experiment", type=str, default="experiment_0")
    train_all_parser.add_argument("--image-size", type=int, default=256)
    train_all_parser.add_argument("--batch-size", type=int, default=16)
    train_all_parser.add_argument("--epochs", type=int, default=10)
    train_all_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_all_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_all_parser.add_argument("--num-workers", type=int, default=0)
    train_all_parser.add_argument("--max-train-samples", type=int, default=400)
    train_all_parser.add_argument("--max-test-samples", type=int, default=200)
    train_all_parser.add_argument("--seed", type=int, default=42)
    train_all_parser.add_argument("--gif-path", type=Path, default=None)
    train_all_parser.add_argument("--accuracy-gif-path", type=Path, default=None)
    train_all_parser.add_argument("--vae-recon-weight", type=float, default=0.35)
    train_all_parser.add_argument("--vae-kl-weight", type=float, default=0.02)
    train_all_parser.add_argument("--force-cpu", action="store_true")

    gif_from_history_parser = subparsers.add_parser(
        "gifs-from-history",
        help="Crea GIFs de loss y accuracy a partir de historiales del mismo experimento.",
    )
    gif_from_history_parser.add_argument(
        "history_paths",
        type=Path,
        nargs="+",
        help="JSONs de historial, por ejemplo cnn_..., mlp_..., kan_...",
    )
    gif_from_history_parser.add_argument("--loss-gif-path", type=Path, default=None)
    gif_from_history_parser.add_argument("--accuracy-gif-path", type=Path, default=None)
    gif_from_history_parser.add_argument("--duration", type=float, default=0.8)

    predict_parser = subparsers.add_parser("predict", help="Predice la clase de una imagen.")
    predict_parser.add_argument("image_path", type=Path)
    predict_parser.add_argument("checkpoint_path", type=Path)
    predict_parser.add_argument("--top-k", type=int, default=3)
    predict_parser.add_argument("--force-cpu", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_model(
            model_type=args.model_type,
            dataset_root=args.dataset_root,
            split_name=args.split_name,
            modality=args.modality,
            experiment=args.experiment,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
            checkpoint_path=args.checkpoint_path,
            history_path=args.history_path,
            vae_recon_weight=args.vae_recon_weight,
            vae_kl_weight=args.vae_kl_weight,
            force_cpu=args.force_cpu,
        )
        return

    if args.command == "train-all":
        train_all_models(
            dataset_root=args.dataset_root,
            split_name=args.split_name,
            modality=args.modality,
            experiment=args.experiment,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            seed=args.seed,
            vae_recon_weight=args.vae_recon_weight,
            vae_kl_weight=args.vae_kl_weight,
            force_cpu=args.force_cpu,
            gif_path=args.gif_path,
            accuracy_gif_path=args.accuracy_gif_path,
        )
        return

    if args.command == "gifs-from-history":
        first_name = args.history_paths[0].stem
        parts = first_name.split("_")
        if len(parts) < 4:
            raise ValueError("No se pudo inferir modalidad/split/experimento desde el nombre del historial.")
        modality = parts[1]
        split_name = parts[2]
        experiment = "_".join(parts[3:])
        loss_gif_path = args.loss_gif_path or default_gif_path(modality, split_name, experiment)
        accuracy_gif_path = args.accuracy_gif_path or default_accuracy_gif_path(
            modality, split_name, experiment
        )
        result = create_gifs_from_history_files(
            history_paths=args.history_paths,
            loss_gif_path=loss_gif_path,
            accuracy_gif_path=accuracy_gif_path,
            duration=args.duration,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "predict":
        predictions = predict_image(
            image_path=args.image_path,
            checkpoint_path=args.checkpoint_path,
            top_k=args.top_k,
            force_cpu=args.force_cpu,
        )
        print(json.dumps(predictions, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Comando no soportado: {args.command}")


if __name__ == "__main__":
    main()
