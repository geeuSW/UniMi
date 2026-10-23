"""Confronta le implementazioni scritte da zero con quelle di scikit-learn sugli stessi dati.
scikit-learn è usato solo qui come riferimento, mai negli esperimenti."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as skm

from src.calibration import IsotonicCalibration
from src.data import load_dataset, train_cal_test_split
from src.metrics import accuracy, brier_score, log_loss
from src.models import make_model


def check(name, ours, ref, atol):
    diff = np.max(np.abs(np.asarray(ours) - np.asarray(ref)))
    status = "OK " if diff < atol else "FAIL"
    print(f"[{status}] {name:<45} differenza massima = {diff:.2e}")


def main():
    X, y = load_dataset("adult")
    split = train_cal_test_split(X, y, seed=0)
    X_tr, y_tr = split["train"]
    X_cal, y_cal = split["cal"]

    # regressione logistica: stessa regolarizzazione (C = 1 / (lambda * n))
    ours = make_model("logreg", seed=0).fit(X_tr, y_tr)
    ref = LogisticRegression(C=1 / (ours.l2 * len(y_tr)), tol=1e-10, max_iter=10000).fit(X_tr, y_tr)
    p_ours, p_ref = ours.predict_proba(X_cal), ref.predict_proba(X_cal)[:, 1]
    check("regressione logistica: probabilità", p_ours, p_ref, 1e-2)

    # regressione isotonica (sui punteggi della LR e della RF)
    rf_scores = make_model("rf", seed=0).fit(X_tr, y_tr).scores(X_cal)
    for name, s in [("LR", ours.scores(X_cal)), ("RF", rf_scores)]:
        iso = IsotonicCalibration().fit(s, y_cal)
        ref_iso = IsotonicRegression(out_of_bounds="clip").fit(s, y_cal)
        check(f"regressione isotonica (punteggi {name})", iso.predict(s), ref_iso.predict(s), 1e-9)

    # metriche
    check("accuracy", accuracy(y_cal, p_ours), skm.accuracy_score(y_cal, p_ours > 0.5), 1e-12)
    check("log-loss", log_loss(y_cal, p_ours), skm.log_loss(y_cal, p_ours), 1e-9)
    check("Brier score", brier_score(y_cal, p_ours), skm.brier_score_loss(y_cal, p_ours), 1e-12)


if __name__ == "__main__":
    main()
