# ============================================================
# GIF: evolución de dos qubits |00> al aplicar Ry(theta)
# en paralelo sobre q0 y q1, con theta desde 0 hasta pi
#
# Circuito:
#   q0: |0> --- Ry(theta) ---
#   q1: |0> --- Ry(theta) ---
#
# Genera:
#   ry_q0_q1_probabilities.gif
#
# Requisitos:
#   pip install numpy matplotlib imageio pillow
# ============================================================

import os
import shutil
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
OUT_DIR = "gif_ry_q0_q1"
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
GIF_PATH = os.path.join(OUT_DIR, "ry_q0_q1_probabilities.gif")

N_FRAMES = 80
FPS = 18
DPI = 130

if os.path.exists(FRAMES_DIR):
    shutil.rmtree(FRAMES_DIR)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Funciones cuánticas
# ------------------------------------------------------------
def ry(theta: float) -> np.ndarray:
    """
    Matriz Ry(theta):
        [[cos(theta/2), -sin(theta/2)],
         [sin(theta/2),  cos(theta/2)]]
    """
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([
        [c, -s],
        [s,  c]
    ], dtype=np.float64)

def system_state(theta: float) -> np.ndarray:
    """
    Estado del sistema tras aplicar Ry(theta) a q0 y q1 en paralelo:
        |psi(theta)> = (Ry(theta) ⊗ Ry(theta)) |00>
    Base computacional: |00>, |01>, |10>, |11>
    """
    ket0 = np.array([1.0, 0.0], dtype=np.float64)
    ket00 = np.kron(ket0, ket0)

    U = np.kron(ry(theta), ry(theta))
    psi = U @ ket00
    return psi

def basis_probabilities(theta: float):
    psi = system_state(theta)
    probs = np.abs(psi) ** 2
    return {
        "|00>": float(probs[0]),
        "|01>": float(probs[1]),
        "|10>": float(probs[2]),
        "|11>": float(probs[3]),
    }

def single_qubit_marginals(theta: float):
    """
    Como las puertas son paralelas e iguales, los marginales de q0 y q1 coinciden:
        P(0) = cos^2(theta/2)
        P(1) = sin^2(theta/2)
    """
    p0 = float(np.cos(theta / 2.0) ** 2)
    p1 = float(np.sin(theta / 2.0) ** 2)
    return p0, p1

# ------------------------------------------------------------
# Utilidades de dibujo
# ------------------------------------------------------------
def draw_circuit(ax, theta: float):
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    y0 = 0.55
    y1 = -0.55

    # Líneas de los qubits
    ax.plot([1.0, 9.2], [y0, y0], linewidth=2)
    ax.plot([1.0, 9.2], [y1, y1], linewidth=2)

    # Etiquetas iniciales
    ax.text(0.15, y0, r"$q_0:\ |0\rangle$", va="center", fontsize=14)
    ax.text(0.15, y1, r"$q_1:\ |0\rangle$", va="center", fontsize=14)

    # Puertas Ry en paralelo
    gate0 = FancyBboxPatch(
        (3.8, y0 - 0.28), 1.9, 0.56,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=2, facecolor="white"
    )
    gate1 = FancyBboxPatch(
        (3.8, y1 - 0.28), 1.9, 0.56,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=2, facecolor="white"
    )
    ax.add_patch(gate0)
    ax.add_patch(gate1)

    ax.text(4.75, y0, r"$R_y(\theta)$", ha="center", va="center", fontsize=14)
    ax.text(4.75, y1, r"$R_y(\theta)$", ha="center", va="center", fontsize=14)

    # Texto theta
    ax.text(6.5, 0.38, rf"$\theta = {theta:.3f}$ rad", fontsize=13)
    ax.text(6.5, -0.02, rf"$\theta/\pi = {theta/np.pi:.3f}$", fontsize=13)

def draw_single_qubit_probs(ax, theta: float):
    p0, p1 = single_qubit_marginals(theta)

    labels = [r"$P(q_0=0)$", r"$P(q_0=1)$", r"$P(q_1=0)$", r"$P(q_1=1)$"]
    vals = [p0, p1, p0, p1]
    xpos = np.arange(len(labels))

    bars = ax.bar(xpos, vals, width=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Probabilidad", fontsize=12)
    ax.set_title("Probabilidades marginales de q0 y q1", fontsize=14)
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11
        )

def draw_system_probs(ax, theta: float):
    probs = basis_probabilities(theta)

    labels = list(probs.keys())
    vals = list(probs.values())
    xpos = np.arange(len(labels))

    bars = ax.bar(xpos, vals, width=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Probabilidad", fontsize=12)
    ax.set_title("Probabilidades de los estados del sistema", fontsize=14)
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11
        )

def draw_theory_plot(ax, theta: float):
    ts = np.linspace(0, np.pi, 400)

    p00 = np.cos(ts / 2.0) ** 4
    p01 = (np.cos(ts / 2.0) ** 2) * (np.sin(ts / 2.0) ** 2)
    p10 = p01
    p11 = np.sin(ts / 2.0) ** 4

    p00_now = np.cos(theta / 2.0) ** 4
    p01_now = (np.cos(theta / 2.0) ** 2) * (np.sin(theta / 2.0) ** 2)
    p11_now = np.sin(theta / 2.0) ** 4

    ax.plot(ts, p00, linewidth=2.2, label=r"$P(00)=\cos^4(\theta/2)$")
    ax.plot(ts, p01, linewidth=2.2, label=r"$P(01)=P(10)=\cos^2(\theta/2)\sin^2(\theta/2)$")
    ax.plot(ts, p11, linewidth=2.2, label=r"$P(11)=\sin^4(\theta/2)$")

    ax.scatter([theta], [p00_now], s=35, zorder=3)
    ax.scatter([theta], [p01_now], s=35, zorder=3)
    ax.scatter([theta], [p11_now], s=35, zorder=3)

    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel("Probabilidad", fontsize=12)
    ax.set_title("Leyes teóricas del sistema", fontsize=14)
    ax.grid(alpha=0.25)

    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"], fontsize=12)
    ax.legend(loc="upper center", fontsize=9)

# ------------------------------------------------------------
# Generación de frames
# ------------------------------------------------------------
thetas = np.linspace(0.0, np.pi, N_FRAMES)

for i, theta in enumerate(thetas):
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.1, 1.15], hspace=0.42, wspace=0.28)

    ax_circuit = fig.add_subplot(gs[0, :])
    ax_single = fig.add_subplot(gs[1, 0])
    ax_system = fig.add_subplot(gs[1, 1])
    ax_theory = fig.add_subplot(gs[2, :])

    draw_circuit(ax_circuit, theta)
    draw_single_qubit_probs(ax_single, theta)
    draw_system_probs(ax_system, theta)
    draw_theory_plot(ax_theory, theta)

    psi = system_state(theta)
    amp_text = (
        r"$|\psi(\theta)\rangle = (R_y(\theta)\otimes R_y(\theta))|00\rangle$" "\n"
        rf"$= {psi[0]:.3f}|00\rangle + {psi[1]:.3f}|01\rangle + {psi[2]:.3f}|10\rangle + {psi[3]:.3f}|11\rangle$"
    )

    fig.suptitle(
        r"Evolución de dos qubits $q_0,q_1$ bajo $R_y(\theta)$ en paralelo, con estado inicial $|00\rangle$",
        fontsize=16
    )
    fig.text(0.5, 0.03, amp_text, ha="center", fontsize=11)

    frame_path = os.path.join(FRAMES_DIR, f"frame_{i:04d}.png")
    plt.savefig(frame_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Crear GIF
# ------------------------------------------------------------
images = []
for i in range(N_FRAMES):
    frame_path = os.path.join(FRAMES_DIR, f"frame_{i:04d}.png")
    images.append(imageio.imread(frame_path))

imageio.mimsave(GIF_PATH, images, fps=FPS)

# Borrar frames temporales
shutil.rmtree(FRAMES_DIR)

print(f"GIF guardado en: {GIF_PATH}")
