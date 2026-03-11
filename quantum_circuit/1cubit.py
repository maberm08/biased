# ============================================================
# GIF: evolución de un qubit |0> al aplicar R_y(theta)
# con theta desde 0 hasta pi
#
# Genera:
#   ry_q0_probabilities.gif
#
# Requisitos:
#   pip install numpy matplotlib imageio pillow
# ============================================================

import os
import shutil
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
OUT_DIR = "gif_ry_q0"
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
GIF_PATH = os.path.join(OUT_DIR, "ry_q0_probabilities.gif")

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
def state_after_ry(theta: float) -> np.ndarray:
    """
    Estado resultante de aplicar R_y(theta) a |0>.

    R_y(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    """
    return np.array([
        np.cos(theta / 2.0),
        np.sin(theta / 2.0)
    ], dtype=np.float64)

def probabilities(theta: float):
    psi = state_after_ry(theta)
    p0 = float(np.abs(psi[0])**2)
    p1 = float(np.abs(psi[1])**2)
    return p0, p1

def bloch_coords(theta: float):
    """
    Para R_y(theta)|0>, el vector de Bloch es:
    x = sin(theta), y = 0, z = cos(theta)
    """
    return np.sin(theta), 0.0, np.cos(theta)

# ------------------------------------------------------------
# Utilidades de dibujo
# ------------------------------------------------------------
def draw_circuit(ax, theta: float):
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # Línea del qubit
    ax.plot([0.8, 9.2], [0, 0], linewidth=2)

    # Etiqueta inicial
    ax.text(0.2, 0, r"$q_0:\ |0\rangle$", va="center", fontsize=14)

    # Caja de la puerta
    gate = FancyBboxPatch(
        (3.6, -0.35), 2.2, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=2,
        facecolor="white"
    )
    ax.add_patch(gate)
    ax.text(4.7, 0.0, rf"$R_y(\theta)$", ha="center", va="center", fontsize=15)

    # Valor actual de theta
    ax.text(7.0, 0.35, rf"$\theta = {theta:.3f}$ rad", fontsize=13)
    ax.text(7.0, -0.10, rf"$\theta/\pi = {theta/np.pi:.3f}$", fontsize=13)

def draw_probabilities(ax, p0: float, p1: float):
    labels = [r"$P(0)$", r"$P(1)$"]
    vals = [p0, p1]
    xpos = np.arange(len(labels))

    bars = ax.bar(xpos, vals, width=0.55)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Probabilidad", fontsize=12)
    ax.set_title("Probabilidades de medida", fontsize=14)
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            val + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=12
        )

def draw_formula_plot(ax, theta: float):
    ts = np.linspace(0, np.pi, 400)
    p1_curve = np.sin(ts / 2.0) ** 2
    p1_now = np.sin(theta / 2.0) ** 2

    ax.plot(ts, p1_curve, linewidth=2.5, label=r"$P(1)=\sin^2(\theta/2)$")
    ax.scatter([theta], [p1_now], s=60, zorder=3)

    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$P(1)$", fontsize=12)
    ax.set_title("Ley teórica", fontsize=14)
    ax.grid(alpha=0.25)

    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"], fontsize=12)
    ax.legend(loc="lower right", fontsize=11)

def draw_bloch_panel(ax, theta: float):
    """
    Representación 2D sencilla del meridiano x-z de la esfera de Bloch.
    """
    x, _, z = bloch_coords(theta)

    # Circunferencia de radio 1
    circle = Circle((0, 0), 1.0, fill=False, linewidth=2)
    ax.add_patch(circle)

    # Ejes
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)

    # Vector actual
    ax.arrow(
        0, 0, x, z,
        head_width=0.06, head_length=0.08,
        length_includes_head=True,
        linewidth=2.5
    )

    ax.text(0, 1.08, r"$|0\rangle$", ha="center", va="bottom", fontsize=12)
    ax.text(0, -1.12, r"$|1\rangle$", ha="center", va="top", fontsize=12)
    ax.text(1.08, 0, r"$x$", ha="left", va="center", fontsize=12)
    ax.text(-1.08, 0, r"$-x$", ha="right", va="center", fontsize=12)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title("Meridiano de Bloch (x-z)", fontsize=14)
    ax.axis("off")

# ------------------------------------------------------------
# Generación de frames
# ------------------------------------------------------------
thetas = np.linspace(0.0, np.pi, N_FRAMES)

for i, theta in enumerate(thetas):
    p0, p1 = probabilities(theta)

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.28)

    ax_circuit = fig.add_subplot(gs[0, :])
    ax_probs = fig.add_subplot(gs[1, 0])
    ax_formula = fig.add_subplot(gs[1, 1])

    draw_circuit(ax_circuit, theta)
    draw_probabilities(ax_probs, p0, p1)
    draw_formula_plot(ax_formula, theta)

    # Eje pequeño extra para Bloch, superpuesto suavemente
    bloch_ax = fig.add_axes([0.70, 0.53, 0.18, 0.22])
    draw_bloch_panel(bloch_ax, theta)

    fig.suptitle(
        r"Evolución del qubit $q_0$ bajo $R_y(\theta)$ con estado inicial $|0\rangle$",
        fontsize=16
    )

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
