# Sentiment Analysis on Tweets 🐦💬

This is a machine learning project for performing sentiment analysis on tweets using the **Sentiment140** dataset. The goal is to classify tweets as **Positive** or **Negative** based on their content.

## 📂 Dataset
- **Sentiment140.csv** — contains 1.6 million tweets labeled as:
  - `0` → Negative
  - `4` → Positive

## ⚙️ Features Used
- **TF-IDF (Term Frequency-Inverse Document Frequency)** for converting text to numerical features.
- **Logistic Regression** as the classification model.

## 🚀 Project Workflow
1. Load and clean the dataset.
2. Preprocess the tweets (remove URLs, mentions, hashtags, special characters).
3. Convert text into TF-IDF features.
4. Train the Logistic Regression model.
5. Evaluate the model using a classification report.

## 📊 Model Performance
| Metric     | Score  |
|------------|--------|
| Accuracy   | ~79%   |
| Precision  | 0.78 - 0.80 |
| Recall     | 0.78 - 0.80 |
| F1-Score   | ~0.79  |
