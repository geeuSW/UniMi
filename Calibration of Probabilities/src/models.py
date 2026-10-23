import numpy as np
from sklearn.ensemble import RandomForestClassifier


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionGD:
    """Regressione logistica addestrata con discesa del gradiente (full-batch) sulla
    log-loss media con regolarizzazione L2 sui pesi (il bias non è regolarizzato)."""

    def __init__(self, learning_rate=1.0, n_iter=5000, l2=1e-3, tol=1e-6):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2 = l2
        self.tol = tol

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for self.n_iter_done in range(1, self.n_iter + 1):
            error = sigmoid(X @ self.w + self.b) - y          # p_i - y_i
            grad_w = X.T @ error / n + self.l2 * self.w       # gradiente rispetto a w
            grad_b = error.mean()                             # gradiente rispetto a b
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b
            if np.sqrt(grad_w @ grad_w + grad_b ** 2) < self.tol:   # arresto anticipato
                break
        return self

    def decision_function(self, X):
        return X @ self.w + self.b                            # logit

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    def scores(self, X):
        # punteggio passato ai calibratori: per la LR usiamo il logit
        return self.decision_function(X)


class RandomForest:
    """Wrapper della random forest di scikit-learn con la stessa interfaccia della LR."""

    def __init__(self, n_estimators=300, seed=0):
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]              # media delle predizioni degli alberi

    def scores(self, X):
        # per la RF l'unico punteggio disponibile è la probabilità predetta
        return self.predict_proba(X)


def make_model(name, seed):
    if name == "logreg":
        return LogisticRegressionGD()
    if name == "rf":
        return RandomForest(seed=seed)
    raise ValueError(f"modello sconosciuto: {name}")
