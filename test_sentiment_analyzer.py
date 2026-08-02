from pathlib import Path

from sentiment_analyzer import clean_text, predict_sentiment, train_model


def test_clean_text_removes_punctuation():
    assert clean_text("I LOVE this!!!") == "i love this"


def test_training_creates_model(tmp_path):
    dataset = Path(__file__).parent / "sample_reviews.csv"
    model_path = tmp_path / "test_model.joblib"

    model, accuracy = train_model(str(dataset), str(model_path), test_size=0.2)

    assert model is not None
    assert accuracy >= 0.0
    assert model_path.exists()
    assert predict_sentiment(model, "I absolutely love this product") in {"positive", "negative", "neutral"}
