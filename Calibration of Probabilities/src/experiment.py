from src.calibration import CALIBRATORS
from src.data import train_cal_test_split
from src.metrics import evaluate
from src.models import make_model

MODELS = ("logreg", "rf")
METHODS = ("uncalibrated", "platt", "isotonic")


def calibrated_predictions(model, cal_set, test_set):
    """Addestra ogni calibratore sul calibration set e restituisce le probabilità
    sul test set per ogni metodo (incluso il modello non calibrato)."""
    s_cal, s_test = model.scores(cal_set[0]), model.scores(test_set[0])
    preds = {"uncalibrated": model.predict_proba(test_set[0])}
    for name, calibrator in CALIBRATORS.items():
        preds[name] = calibrator().fit(s_cal, cal_set[1]).predict(s_test)
    return preds


def run_dataset(name, X, y, seeds):
    """Per ogni split casuale: addestra i modelli, li calibra e calcola le metriche sul test set.
    Restituisce le righe di risultati e le predizioni del primo split (per i grafici)."""
    rows, examples = [], {}
    for seed in seeds:
        split = train_cal_test_split(X, y, seed)
        y_test = split["test"][1]
        for model_name in MODELS:
            model = make_model(model_name, seed).fit(*split["train"])
            preds = calibrated_predictions(model, split["cal"], split["test"])
            for method, p in preds.items():
                rows.append({"dataset": name, "model": model_name, "seed": seed, "method": method,
                             **evaluate(y_test, p)})
            if seed == seeds[0]:
                examples[model_name] = {"y_test": y_test, "preds": preds}
        print(f"  {name}: seed {seed} completato")
    return rows, examples
