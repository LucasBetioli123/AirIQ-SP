"""Estilo único dos gráficos (marcas finas, grade discreta, paleta categórica fixa)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ordem fixa (nunca ciclada): azul, laranja, água; cinza para baselines
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
TEXT, MUTED = "#0b0b0b", "#52514e"
SEQ_PM = "Oranges"      # magnitude: um só matiz, claro -> escuro
SEQ_UNC = "Purples"


def apply():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e7e6e2", "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, AQUA, GREY]),
    })


apply()
