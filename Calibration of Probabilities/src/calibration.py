import numpy as np

from src.models import LogisticRegressionGD


class PlattScaling:
    """P(y=1 | f) = 1 / (1 + exp(A f + B)), con A e B stimati sul calibration set
    minimizzando la log-loss. In pratica è una regressione logistica con una sola feature:
    il punteggio f del modello."""

    def __init__(self, learning_rate=1.0, n_iter=10000, tol=1e-8):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, scores, y):
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        targets = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

        # standardizzo il punteggio, così la discesa del gradiente con passo fisso
        # funziona sia con i logit della LR sia con le probabilità della RF
        self.mu, self.sigma = scores.mean(), scores.std()
        z = ((scores - self.mu) / self.sigma).reshape(-1, 1)
        lr = LogisticRegressionGD(self.learning_rate, self.n_iter, l2=0.0, tol=self.tol).fit(z, targets)

        # riporto i parametri alla scala originale e alla convenzione di segno del paper
        self.A = -lr.w[0] / self.sigma
        self.B = -(lr.b - lr.w[0] * self.mu / self.sigma)
        return self

    def predict(self, scores):
        return 1.0 / (1.0 + np.exp(np.clip(self.A * scores + self.B, -500, 500)))


class IsotonicCalibration:
    """Regressione isotonica con l'algoritmo Pool Adjacent Violators (PAV).
    Il risultato è una funzione a gradini non decrescente del punteggio."""

    @staticmethod
    def _round(scores):
        # punteggi uguali a meno di errori di arrotondamento vanno trattati come pareggi
        return np.round(scores, 10)

    def fit(self, scores, y):
        scores = self._round(scores)
        order = np.argsort(scores, kind="mergesort")
        s, t = scores[order], y[order].astype(float)

        # un blocco iniziale per ogni punteggio distinto (valore = media delle etichette, peso = numero di esempi)
        unique_scores, start = np.unique(s, return_index=True)
        counts = np.diff(np.append(start, len(s)))
        sums = np.add.reduceat(t, start)

        # scorro i blocchi e fondo quelli adiacenti che violano l'ordine (media pesata)
        means, weights, upper = [], [], []
        for m, w, u in zip(sums / counts, counts, unique_scores):
            means.append(m)
            weights.append(w)
            upper.append(u)
            while len(means) > 1 and means[-2] >= means[-1]:
                w_new = weights[-2] + weights[-1]
                m_new = (weights[-2] * means[-2] + weights[-1] * means[-1]) / w_new
                means[-2:] = [m_new]
                weights[-2:] = [w_new]
                upper[-2:] = [upper[-1]]

        self.thresholds = np.array(upper)   # estremo superiore di ogni gradino
        self.values = np.array(means)       # valore di ogni gradino
        return self

    def predict(self, scores):
        idx = np.searchsorted(self.thresholds, self._round(scores), side="left")
        return self.values[np.minimum(idx, len(self.values) - 1)]


CALIBRATORS = {"platt": PlattScaling, "isotonic": IsotonicCalibration}
