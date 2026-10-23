"""Module for text preprocessing of news headlines."""

import re
import string


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so",
    "of", "to", "in", "on", "at", "by", "for", "with", "about",
    "as", "into", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might",
    "this", "that", "these", "those", "it", "its", "new"
}


def clean_text(text):
    """Clean a single headline: lowercase, remove punctuation and stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)                     # remove digits
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def preprocess_dataframe(df, text_column="headline"):
    """Apply cleaning to the text column and add a 'clean_headline' column."""
    df = df.copy()
    df["clean_headline"] = df[text_column].astype(str).apply(clean_text)
    print("Preprocessing completed.")
    print(f"Sample cleaned headline: '{df['clean_headline'].iloc[0]}'")
    return df