# Calibrazione di classificatori probabilistici

Progetto per il corso *Statistical Methods for Machine Learning* (Assignment 2), basato sull'articolo
di A. Niculescu-Mizil e R. Caruana, *Predicting Good Probabilities with Supervised Learning*, ICML 2005.

Si valuta e si corregge la calibrazione di due classificatori, una regressione logistica (implementata
da zero) e una random forest (scikit-learn), su due dataset reali (Adult e MAGIC Gamma Telescope),
usando Platt scaling e regressione isotonica, entrambi implementati da zero.

Il report è in [`report/report.pdf`](report/report.pdf).

## Struttura

```
main.py              esegue gli esperimenti, salva i risultati e produce le figure
sanity_checks.py     confronta le implementazioni da zero con scikit-learn
src/
  data.py            download dei dataset, preprocessing, split training/calibrazione/test
  models.py          regressione logistica (discesa del gradiente) e wrapper della random forest
  calibration.py     Platt scaling e regressione isotonica (algoritmo PAV)
  metrics.py         accuracy, log-loss, Brier score, ECE, reliability curve
  experiment.py      pipeline sperimentale
  plots.py           reliability diagram
results/             CSV e figure generati da main.py
report/              sorgente LaTeX e PDF del report
```

## Requisiti

Serve **Python 3.12 o superiore**: `numpy==2.5.3` non è disponibile per versioni precedenti, e
`pip install` con Python 3.11 o inferiore fallisce con `No matching distribution found for numpy==2.5.3`.
Le versioni esatte delle librerie sono fissate in [`requirements.txt`](requirements.txt):

| Libreria     | Versione |
|--------------|----------|
| numpy        | 2.5.3    |
| pandas       | 3.0.6    |
| scikit-learn | 1.9.1    |
| matplotlib   | 3.11.2   |

Per controllare le versioni di Python installate: `python3 --version`, `python3.12 --version`.

## Installazione

Si crea un ambiente virtuale con Python 3.12 e vi si installano le librerie.

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version                  # deve stampare Python 3.12.x o superiore
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
pip install --upgrade pip
pip install -r requirements.txt
```

Se esiste già una cartella `.venv` creata con una versione di Python precedente, va cancellata
prima di ricrearla.

In alternativa, con conda:

```bash
conda create -n calibrazione python=3.12
conda activate calibrazione
pip install -r requirements.txt
```

## Esecuzione

Con l'ambiente attivo:

```bash
python main.py            # circa un minuto; alla prima esecuzione scarica i dataset in data/
python sanity_checks.py   # opzionale
```

`python main.py --seeds 1` esegue un solo split per una prova veloce.

I dataset sono scaricati dal repository [PMLB](https://github.com/EpistasisLab/pmlb), che
ridistribuisce i dataset UCI originali ([Adult](https://archive.ics.uci.edu/dataset/2/adult),
[MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)).