"""
clustering_3d_visualization.py
------------------------------
3D PCA-based visualization of K-Means behavioral customer clusters.

Purpose:
- Visualize high-dimensional customer behavior in 3D
- Interpret behavioral clusters discovered by K-Means
- Independent of CLV prediction & classification
"""

import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def visualize_kmeans_clusters_3d(
    data_path: str = "data/customers.csv",
    n_clusters: int = 3
):
    # ── Load customer dataset ─────────────────────────
    df = pd.read_csv(data_path)

    # ── Behavioral features used for clustering ───────
    behavioral_features = [
        "Frequency",
        "AvgSpend",
        "Recency",
        "Tenure",
        "RFM_Score"
    ]

    X = df[behavioral_features]

    # ── Standardize features ─────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── K-Means clustering ───────────────────────────
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # ── PCA (3D) ─────────────────────────────────────
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]
    df["PCA3"] = X_pca[:, 2]

    # ── 3D Visualization ─────────────────────────────
    fig = px.scatter_3d(
        df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Cluster",
        title="3D Customer Behavioral Clustering (K-Means + PCA)",
        labels={
            "PCA1": "Principal Component 1",
            "PCA2": "Principal Component 2",
            "PCA3": "Principal Component 3"
        },
        opacity=0.7,
        hover_data={
            "Frequency": True,
            "AvgSpend": True,
            "Recency": True,
            "Tenure": True,
            "RFM_Score": True,
            "Cluster": True
        }
    )

    fig.update_traces(marker=dict(size=3))
    fig.show()

    # ── Cluster summary (optional) ───────────────────
    summary = (
        df.groupby("Cluster")[behavioral_features]
        .mean()
        .round(2)
    )

    print("\nCluster Behavioral Summary:")
    print(summary)


if __name__ == "__main__":
    visualize_kmeans_clusters_3d()