"""Module for converting text to TF-IDF vectors."""

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_matrix(texts, max_features=500, ngram_range=(1, 2)):
    """Convert list of texts into a TF-IDF matrix."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.9
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return tfidf_matrix, vectorizer