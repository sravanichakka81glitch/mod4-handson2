"""Module for visualizations."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_clusters(matrix, labels, output_path="outputs/cluster_visualization.png"):
    """Reduce TF-IDF to 2D with PCA and scatter-plot clusters."""
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(dense)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=reduced[:, 0], y=reduced[:, 1],
        hue=labels, palette="tab10", s=100
    )
    plt.title("News Headlines — K-Means Clusters (PCA projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Cluster visualization saved to {output_path}")