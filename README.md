# 😊 Emotion Classification

A text-based emotion classification web app built with TF-IDF vectorization and Logistic Regression, deployed on Render.

## 🔴 Live Demo

👉 [Try it here](https://emotion-classification-project-14.onrender.com/)

> Note: Free-tier Render hosting — first load may take 30-50 seconds while the server wakes up.

## 🧠 How It Works

The model classifies input text into emotion categories using a classical NLP pipeline:

1. **Text preprocessing** — cleaning, tokenization
2. **TF-IDF Vectorization** — converts text into numerical feature vectors based on term frequency
3. **Logistic Regression** — multi-class classifier trained on the vectorized features

This is a baseline NLP approach — lightweight, fast to train, and interpretable, though it doesn't capture word order or context the way sequence models (like LSTM or transformers) do.

## 🛠️ Tech Stack

Python · Scikit-learn · Flask · TF-IDF · Render

## 📁 Repository Structure

```
├── data/                  # Training/validation/test datasets
├── notebook/              # Model training notebook
├── templates/             # Frontend HTML
├── app.py                 # Flask backend
├── requirements.txt
├── train.txt / val.txt / test.txt
```

## 🚧 Known Limitations

- TF-IDF + Logistic Regression treats text as a bag of words — it doesn't understand context, sarcasm, or word order
- Accuracy is bounded by this limitation compared to deep learning approaches (LSTM, BERT)
- Planned upgrade: replace with a sequence model (LSTM/transformer-based) for better contextual understanding

## 👤 Author

**Ahad Ahmad**
- GitHub: [@AhadAhmad0](https://github.com/AhadAhmad0)

