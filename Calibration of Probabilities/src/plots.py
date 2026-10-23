import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import reliability_curve

STYLE = {
    "uncalibrated": {"color": "#2a78d6", "marker": "o", "label": "Non calibrato"},
    "platt": {"color": "#eb6834", "marker": "s", "label": "Platt scaling"},
    "isotonic": {"color": "#1baf7a", "marker": "^", "label": "Regressione isotonica"},
}
MODEL_NAMES = {"logreg": "Regressione logistica", "rf": "Random forest"}
DATASET_NAMES = {"adult": "Adult", "magic": "MAGIC"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def plot_reliability(examples, dataset, path, n_bins=10):
    """Un reliability diagram per modello, con le curve prima e dopo la calibrazione."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for ax, (model_name, ex) in zip(axes, examples.items()):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Calibrazione perfetta")
        for method, p in ex["preds"].items():
            mean_p, frac_pos, _ = reliability_curve(ex["y_test"], p, n_bins)
            st = STYLE[method]
            ax.plot(mean_p, frac_pos, marker=st["marker"], color=st["color"], label=st["label"],
                    linewidth=2, markersize=6, markeredgecolor="white", markeredgewidth=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=MODEL_NAMES[model_name],
               xlabel="Probabilità predetta media", ylabel="Frazione di positivi")
        ax.set_aspect("equal")
        ax.grid(color="#e1e0d9", linewidth=0.6)
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(f"{DATASET_NAMES[dataset]} - test set", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
