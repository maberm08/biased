# ============================================================
# COMPARACIÓN EN GIF DE 3 CLASIFICADORES MONONEURONALES 1D
#   1) Perceptrón
#   2) Neurona KAN
#   3) Cubit (single-qubit classifier simulado)
#
# Los datos son unidimensionales:
#   cada par (x, y) significa:
#       x = entrada
#       y = etiqueta
#
# Se generan dos GIFs:
#   - gif_ejercicio_1_1d.gif
#   - gif_ejercicio_2_1d.gif
#
# Requisitos:
#   pip install torch matplotlib imageio pillow numpy
# ============================================================

import os
import math
import shutil
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cpu"
DTYPE = torch.float32

EPOCHS = 1000
LR = 0.03
SNAPSHOT_EVERY = 5
FPS = 12

OUT_DIR = "gif_compare_1d_mononeurons"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------
def get_datasets():
    # Ejercicio 1:
    # (1,0),(2,0) clase 0 ; (4,1),(5,1) clase 1
    X1 = np.array([1.0, 2.0, 4.0, 5.0], dtype=np.float32).reshape(-1, 1)
    y1 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    # Ejercicio 2:
    # (1,0),(2,0),(4,0),(5,0) clase 0 ; (3,1),(4,1) clase 1
    X2 = np.array([1.0, 2.0, 5.0, 6.0, 3.0, 4.0], dtype=np.float32).reshape(-1, 1)
    y2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    # Ejercicio 3:
    # (1,0),(2,0),(4,0),(5,0) clase 0 ; (3,1),(4,1) clase 1
    X3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32).reshape(-1, 1)
    y3 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    return [
        ("ejercicio_1_1d", X1, y1),
        ("ejercicio_2_1d", X2, y2),
        ("ejercicio_3_1d", X3, y3),
    ]

# ------------------------------------------------------------
# Normalización
# ------------------------------------------------------------
class Normalizer1D:
    def __init__(self, X):
        self.mu = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0

    def encode(self, X):
        return (X - self.mu) / self.std

# ------------------------------------------------------------
# MODELO 1: Perceptrón mononeuronal 1D
# ------------------------------------------------------------
class Perceptron1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def logits(self, x):
        return self.linear(x).squeeze(-1)

    def forward(self, x):
        return torch.sigmoid(self.logits(x))

# ------------------------------------------------------------
# MODELO 2: KAN mononeuronal 1D
# Una sola función univariante aprendible phi(x), y luego sigmoide
# ------------------------------------------------------------
class PiecewiseLinear1D(nn.Module):
    def __init__(self, xmin=-3.5, xmax=3.5, n_knots=21):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.n_knots = n_knots
        self.delta = (xmax - xmin) / (n_knots - 1)

        self.values = nn.Parameter(torch.zeros(n_knots))
        nn.init.normal_(self.values, mean=0.0, std=0.08)

    def forward(self, x):
        # x: (N,)
        t = (x - self.xmin) / self.delta
        i0 = torch.floor(t).long()
        alpha = t - i0.float()

        i0 = torch.clamp(i0, 0, self.n_knots - 2)
        i1 = i0 + 1

        v0 = self.values[i0]
        v1 = self.values[i1]
        return (1 - alpha) * v0 + alpha * v1

class KAN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = PiecewiseLinear1D()
        self.bias = nn.Parameter(torch.zeros(1))

    def logits(self, x):
        return self.phi(x[:, 0]) + self.bias

    def forward(self, x):
        return torch.sigmoid(self.logits(x))

# ------------------------------------------------------------
# MODELO 3: Cubit 1D
# Clasificador monoqbit simple y diferenciable
# ------------------------------------------------------------
class Cubit1D(nn.Module):
    def __init__(self):
        super().__init__()

        # theta(x) = ax + bx
        self.ax = nn.Parameter(torch.tensor(1.0))
        self.bx = nn.Parameter(torch.tensor(0.0))

    def logits(self, x):
        # x: (N,1)
        theta = self.ax * x[:, 0] + self.bx

        # P(1) = sin^2(theta/2)
        p1 = torch.sin(theta / 2.0) ** 2
        p1 = torch.clamp(p1, 1e-6, 1 - 1e-6)

        return torch.log(p1 / (1 - p1))

    def forward(self, x):
        theta = self.ax * x[:, 0] + self.bx
        p1 = torch.sin(theta / 2.0) ** 2
        return torch.clamp(p1, 1e-6, 1 - 1e-6)
# ------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------
def train_models(X_raw, y_raw):
    normalizer = Normalizer1D(X_raw)
    Xn = normalizer.encode(X_raw)

    # Perceptrón y KAN usan entrada normalizada
    X = torch.tensor(Xn, dtype=DTYPE, device=DEVICE)

    # Cubit usa la entrada original, sin normalizar
    X_cubit = torch.tensor(X_raw, dtype=DTYPE, device=DEVICE)

    y = torch.tensor(y_raw, dtype=DTYPE, device=DEVICE)

    models = {
        "Perceptron": Perceptron1D().to(DEVICE),
        "KAN": KAN1D().to(DEVICE),
        "Cubit": Cubit1D().to(DEVICE),
    }

    opts = {
        name: torch.optim.Adam(model.parameters(), lr=LR)
        for name, model in models.items()
    }

    history = {name: [] for name in models.keys()}

    for epoch in range(EPOCHS + 1):
        for name, model in models.items():
            model.train()
            opts[name].zero_grad()

            X_used = X_cubit if name == "Cubit" else X

            logits = model.logits(X_used)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opts[name].step()

        if epoch % SNAPSHOT_EVERY == 0 or epoch == EPOCHS:
            for name, model in models.items():
                model.eval()
                with torch.no_grad():
                    X_used = X_cubit if name == "Cubit" else X
                    logits = model.logits(X_used)
                    loss = F.binary_cross_entropy_with_logits(logits, y).item()
                    probs_train = model(X_used).cpu().numpy()

                history[name].append({
                    "epoch": epoch,
                    "loss": loss,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "probs_train": probs_train,
                })

    return models, history, normalizer
def load_snapshot(model, snapshot):
    model.load_state_dict(snapshot["state_dict"])

def predict_curve(model, model_name, normalizer, x_min, x_max, num=500):
    xs_raw = np.linspace(x_min, x_max, num=num, dtype=np.float32).reshape(-1, 1)

    with torch.no_grad():
        if model_name == "Cubit":
            xt = torch.tensor(xs_raw, dtype=DTYPE, device=DEVICE)
        else:
            xs_norm = normalizer.encode(xs_raw)
            xt = torch.tensor(xs_norm, dtype=DTYPE, device=DEVICE)

        probs = model(xt).cpu().numpy()

    return xs_raw[:, 0], probs
# ------------------------------------------------------------
# Visualización y GIF
# ------------------------------------------------------------
def make_gif(dataset_name, X_raw, y_raw):
    models, history, normalizer = train_models(X_raw, y_raw)

    x_min = float(X_raw.min()) - 1.0
    x_max = float(X_raw.max()) + 1.0

    frames_dir = os.path.join(OUT_DIR, f"frames_{dataset_name}")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    n_frames = len(history["Perceptron"])

    colors = {
        "Perceptron": "tab:blue",
        "KAN": "tab:green",
        "Cubit": "tab:red",
    }

    for k in range(n_frames):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        # curvas
        legend_handles = []
        for name, model in models.items():
            load_snapshot(model, history[name][k])
            xs, probs = predict_curve(model, name, normalizer, x_min, x_max, num=600)
            ax.plot(xs, probs, linewidth=2.7, color=colors[name], label=name)
            legend_handles.append(
                plt.Line2D([0], [0], color=colors[name], lw=2.7, label=name)
            )

        # puntos del dataset en y=0 o y=1
        X0 = X_raw[y_raw == 0][:, 0]
        X1 = X_raw[y_raw == 1][:, 0]

        ax.scatter(X0, np.zeros_like(X0), s=95, marker="o", edgecolor="k", zorder=5, label="Clase 0")
        ax.scatter(X1, np.ones_like(X1), s=110, marker="^", edgecolor="k", zorder=5, label="Clase 1")

        # línea p=0.5
        ax.axhline(0.5, linestyle="--", linewidth=1.2, alpha=0.6)

        ep = history["Perceptron"][k]["epoch"]
        lp = history["Perceptron"][k]["loss"]
        lk = history["KAN"][k]["loss"]
        lc = history["Cubit"][k]["loss"]

        ax.set_title(
            f"{dataset_name} | época {ep}\n"
            f"Loss -> Perceptrón: {lp:.4f} | KAN: {lk:.4f} | Cubit: {lc:.4f}",
            fontsize=12
        )
        ax.set_xlabel("x")
        ax.set_ylabel("probabilidad predicha de clase 1")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.08, 1.08)
        ax.grid(alpha=0.25)

        point_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='C0',
                       markeredgecolor='k', markersize=9, label='Clase 0'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='C1',
                       markeredgecolor='k', markersize=10, label='Clase 1'),
        ]

        ax.legend(handles=legend_handles + point_handles, loc="best")

        frame_path = os.path.join(frames_dir, f"frame_{k:04d}.png")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=120)
        plt.close(fig)

    gif_path = os.path.join(OUT_DIR, f"gif_{dataset_name}.gif")
    images = []
    for k in range(n_frames):
        images.append(imageio.imread(os.path.join(frames_dir, f"frame_{k:04d}.png")))
    imageio.mimsave(gif_path, images, fps=FPS)

    shutil.rmtree(frames_dir)

    print(f"GIF guardado en: {gif_path}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    datasets = get_datasets()
    for dataset_name, X_raw, y_raw in datasets:
        make_gif(dataset_name, X_raw, y_raw)

    print("\nTerminado.")
