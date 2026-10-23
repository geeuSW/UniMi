import argparse
import os

import pandas as pd

from src.data import DATASETS, load_dataset
from src.experiment import run_dataset
from src.plots import plot_reliability

SEEDS = [0, 1, 2, 3, 4]
OUT_DIR = "results"
METRICS = ["accuracy", "log_loss", "brier", "ece"]


def main(seeds):
    os.makedirs(f"{OUT_DIR}/figures", exist_ok=True)

    rows = []
    for name in DATASETS:
        print(f"Esecuzione su {name}")
        X, y = load_dataset(name)
        dataset_rows, examples = run_dataset(name, X, y, seeds)
        rows += dataset_rows
        plot_reliability(examples, name, f"{OUT_DIR}/figures/reliability_{name}.pdf")

    # una riga per dataset / modello / metodo / seed
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/metrics_per_seed.csv", index=False)

    # media e deviazione standard sui seed
    g = df.groupby(["dataset", "model", "method"], sort=False)[METRICS]
    summary = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).reset_index()
    summary.to_csv(f"{OUT_DIR}/summary.csv", index=False)

    print()
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 4):
        print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="numero di split casuali (max 5)")
    args = parser.parse_args()
    main(SEEDS[:args.seeds])
