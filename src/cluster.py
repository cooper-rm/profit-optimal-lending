"""KMeans cluster features — distance from each row to every cluster centroid.

Unsupervised (never sees the target), so it can be fit on all rows without label leakage.
The per-centroid distances are fed back in as extra columns for the supervised models, and
the fitted clusters double as the segments for the regime analysis.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def kmeans_features(X, k=8, seed=0):
    """Return a DataFrame of distances from each row of ``X`` to each of ``k`` cluster centroids."""
    Z = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X)
    dist = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_transform(Z)
    return pd.DataFrame(dist, columns=[f"kmeans_d{i}" for i in range(k)], index=X.index)
