"""Entry point: Clustering News Headlines Using K-Means."""

import os
import pandas as pd

from src.data_loader import load_data, preview_data
from src.preprocessor import preprocess_dataframe
from src.vectorizer import build_tfidf_matrix
from src.clusterer import find_optimal_k, run_kmeans, get_top_terms_per_cluster
from src.visualizer import plot_clusters


DATA_PATH = "data/news_headlines.csv"
OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load
    df = load_data(DATA_PATH)
    preview_data(df)

    # 2. Preprocess
    df = preprocess_dataframe(df, text_column="headline")

    # Drop empty cleaned rows
    df = df[df["clean_headline"].str.strip() != ""].reset_index(drop=True)

    # 3. Vectorize
    tfidf_matrix, vectorizer = build_tfidf_matrix(df["clean_headline"])

    # 4. Find optimal K (optional) & run clustering
    best_k, _ = find_optimal_k(tfidf_matrix, k_range=range(2, 8))
    model, labels = run_kmeans(tfidf_matrix, n_clusters=best_k)

    # 5. Top terms per cluster
    cluster_terms = get_top_terms_per_cluster(model, vectorizer)
    print("\n--- Top Terms Per Cluster ---")
    for cid, terms in cluster_terms.items():
        print(f"Cluster {cid}: {', '.join(terms)}")

    # 6. Save results
    df["cluster"] = labels
    output_csv = os.path.join(OUTPUT_DIR, "clustered_headlines.csv")
    df[["id", "headline", "clean_headline", "cluster"]].to_csv(
        output_csv, index=False
    )
    print(f"\nClustered results saved to {output_csv}")

    # 7. Visualize
    plot_clusters(
        tfidf_matrix, labels,
        output_path=os.path.join(OUTPUT_DIR, "cluster_visualization.png")
    )

    # 8. Print sample headlines per cluster
    print("\n--- Sample Headlines Per Cluster ---")
    for cid in sorted(df["cluster"].unique()):
        print(f"\nCluster {cid}:")
        for h in df[df["cluster"] == cid]["headline"].head(3):
            print(f"  • {h}")


if __name__ == "__main__":
    main()