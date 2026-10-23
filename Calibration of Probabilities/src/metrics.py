import numpy as np

EPS = 1e-15


def accuracy(y, p):
    return np.mean((p > 0.5) == y)


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_score(y, p):
    return np.mean((p - y) ** 2)


def reliability_curve(y, p, n_bins=10):
    """Bin di uguale ampiezza su [0, 1]. Per ogni bin non vuoto restituisce la probabilità
    predetta media, la frazione di positivi osservata e il numero di esempi."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    sum_y = np.bincount(bins, weights=y, minlength=n_bins)
    mask = counts > 0
    return sum_p[mask] / counts[mask], sum_y[mask] / counts[mask], counts[mask]


def expected_calibration_error(y, p, n_bins=10):
    mean_p, frac_pos, counts = reliability_curve(y, p, n_bins)
    return np.sum(counts * np.abs(mean_p - frac_pos)) / counts.sum()


METRICS = {
    "accuracy": accuracy,
    "log_loss": log_loss,
    "brier": brier_score,
    "ece": expected_calibration_error,
}


def evaluate(y, p):
    return {name: f(y, p) for name, f in METRICS.items()}
