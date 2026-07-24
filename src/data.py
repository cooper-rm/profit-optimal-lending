from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

BLOCKS = ["bureau", "previous_application", "pos_cash_balance", "installments_payments", "credit_card_balance"]


def load_master():
    labels = pd.read_csv(RAW / "application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
    master = labels.merge(pd.read_pickle(INTERIM / "application.pkl"), on="SK_ID_CURR", how="left")
    for name in BLOCKS:
        master = master.merge(pd.read_pickle(INTERIM / f"{name}.pkl"), on="SK_ID_CURR", how="left")
    master = master.set_index("SK_ID_CURR")
    X, y = master.drop(columns="TARGET"), master["TARGET"]
    X.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", c) for c in X.columns]
    return X, y


def load_selected(k=8, rebuild=False):
    """Selected features (+ KMeans distances) and the target, cached to disk.

    Rebuilding the full master just to keep ~350 columns merges several GB of pickles and
    can exhaust memory, so the reduced matrix is cached. Pass ``rebuild=True`` (or delete
    ``model_matrix.pkl``) after re-running feature engineering or selection.
    """
    from .cluster import kmeans_features

    path = INTERIM / "model_matrix.pkl"
    if path.exists() and not rebuild:
        d = pd.read_pickle(path)
        return d.drop(columns="TARGET"), d["TARGET"]
    X, y = load_master()
    kept = pd.read_csv(INTERIM / "null_importance_features.csv")["feature"].tolist()
    X = X[[c for c in kept if c in X.columns]]
    X = pd.concat([X, kmeans_features(X, k=k)], axis=1)
    pd.concat([X, y.rename("TARGET")], axis=1).to_pickle(path)
    return X, y
