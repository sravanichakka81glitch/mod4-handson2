"""Module for K-Means clustering."""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_k(matrix, k_range=range(2, 8)):
    """Evaluate silhouette scores for a range of k values."""
    scores = {}
    for k in k_range:
        if k >= matrix.shape[0]:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        score = silhouette_score(matrix, labels)
        scores[k] = score
        print(f"k={k} -> silhouette score = {score:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"\nOptimal number of clusters: {best_k}")
    return best_k, scores


def run_kmeans(matrix, n_clusters=5):
    """Fit K-Means and return model + labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    print(f"K-Means fitted with {n_clusters} clusters.")
    return model, labels


def get_top_terms_per_cluster(model, vectorizer, n_terms=8):
    """Extract top keywords describing each cluster."""
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for i, center in enumerate(model.cluster_centers_):
        top_idx = center.argsort()[::-1][:n_terms]
        cluster_terms[i] = [terms[j] for j in top_idx]
    return cluster_terms