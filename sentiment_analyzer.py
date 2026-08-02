import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(value):
    if pd.isna(value):
        return "neutral"
    text = str(value).strip().lower()
    if text in {"positive", "pos", "good", "happy", "excellent", "great", "amazing", "fantastic", "love", "superb", "outstanding"}:
        return "positive"
    if text in {"negative", "neg", "bad", "sad", "angry", "hate", "awful", "poor", "terrible", "dislike", "disappointed", "worst"}:
        return "negative"
    return "neutral"


def load_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)

    if "text" in df.columns:
        text_col = "text"
    elif "review" in df.columns:
        text_col = "review"
    elif "comment" in df.columns:
        text_col = "comment"
    else:
        raise ValueError("Dataset must contain a text column named 'text', 'review', or 'comment'.")

    if "label" in df.columns:
        label_col = "label"
    elif "sentiment" in df.columns:
        label_col = "sentiment"
    elif "target" in df.columns:
        label_col = "target"
    else:
        raise ValueError("Dataset must contain a label column named 'label', 'sentiment', or 'target'.")

    data = df[[text_col, label_col]].copy()
    data.columns = ["text", "label"]
    data["text"] = data["text"].fillna("").astype(str).apply(clean_text)
    data["label"] = data["label"].apply(normalize_label)
    data = data[data["text"].str.len() > 0]
    return data["text"], data["label"]


def build_model():
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000, strip_accents="unicode")),
            ("classifier", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
        ]
    )


def train_model(dataset_path, model_path, test_size=0.2):
    texts, labels = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, digits=3))

    model_path = Path(model_path)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model, accuracy


def predict_sentiment(model, text):
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])[0]
    return prediction


def main():
    parser = argparse.ArgumentParser(description="Train and use a sentiment analysis model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--dataset", required=True, help="Path to the CSV file with text and label columns")
    train_parser.add_argument("--model", default="sentiment_model.joblib", help="File name or path for the trained model")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for testing")

    predict_parser = subparsers.add_parser("predict", help="Predict sentiment for a single text")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model")
    predict_parser.add_argument("--text", required=True, help="Text to classify")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.dataset, args.model, args.test_size)
    elif args.command == "predict":
        model = joblib.load(args.model)
        print(predict_sentiment(model, args.text))


if __name__ == "__main__":
    main()
