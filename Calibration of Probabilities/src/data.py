import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
PMLB_URL = "https://github.com/EpistasisLab/pmlb/raw/master/datasets/{name}/{name}.tsv.gz"

DATASETS = {
    "adult": {
        "categorical": ["workclass", "marital-status", "occupation", "relationship", "race", "native-country"],
        "drop": ["fnlwgt", "education"],
        "log": ["capital-gain", "capital-loss"],
        # PMLB codifica '<=50K' come 1: invertiamo in modo che la classe positiva sia '>50K'
        "flip_target": True,
    },
    "magic": {
        "categorical": [],
        "drop": [],
        "log": [],
        "flip_target": False,
    },
}


def download(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.tsv.gz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(PMLB_URL.format(name=name), path)
    return path


def load_dataset(name):
    spec = DATASETS[name]
    df = pd.read_csv(download(name), sep="\t")
    y = df.pop("target").to_numpy().astype(int)
    if spec["flip_target"]:
        y = 1 - y
    X = df.drop(columns=spec["drop"])
    for col in spec["log"]:
        X[col] = np.log1p(X[col])
    X[spec["categorical"]] = X[spec["categorical"]].astype(int).astype(str)
    return X, y


class Preprocessor:
    """One-hot encoding delle colonne categoriche e standardizzazione di quelle numeriche.
    Tutte le statistiche sono calcolate sui dati passati a fit (solo il training set)."""

    def fit(self, X):
        self.cat_cols = X.select_dtypes(exclude="number").columns
        self.num_cols = X.select_dtypes(include="number").columns
        self.mean = X[self.num_cols].mean()
        self.std = X[self.num_cols].std().replace(0, 1)
        self.columns = self._encode(X).columns
        return self

    def _encode(self, X):
        num = (X[self.num_cols] - self.mean) / self.std
        if len(self.cat_cols) == 0:
            return num
        cat = pd.get_dummies(X[self.cat_cols], dtype=float)
        return pd.concat([num, cat], axis=1)

    def transform(self, X):
        return self._encode(X).reindex(columns=self.columns, fill_value=0.0).to_numpy(dtype=float)


def train_cal_test_split(X, y, seed, cal_frac=0.2, test_frac=0.2):
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_rest, y_rest, test_size=cal_frac / (1 - test_frac), stratify=y_rest, random_state=seed)

    prep = Preprocessor().fit(X_train)
    return {
        "train": (prep.transform(X_train), y_train),
        "cal": (prep.transform(X_cal), y_cal),
        "test": (prep.transform(X_test), y_test),
    }
