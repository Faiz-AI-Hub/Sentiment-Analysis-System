# Sentiment Analysis System

A beginner-friendly sentiment analysis project that trains a text classifier to predict whether a review is positive, negative, or neutral.

## Features

- Text preprocessing and cleaning
- Training a machine learning model from a CSV dataset
- CLI commands for training and single-text prediction
- Simple test coverage

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python sentiment_analyzer.py train --dataset sample_reviews.csv --model sentiment_model.joblib
   ```
3. Predict a sentiment:
   ```bash
   python sentiment_analyzer.py predict --model sentiment_model.joblib --text "I love this product!"
   ```

## Dataset Format

The sample dataset uses two columns:

- `text`: the review or sentence
- `label`: `positive`, `negative`, or `neutral`
