Emotion Classification Web App:

This project is an end-to-end Natural Language Processing (NLP) application that classifies human emotions from text input. It uses TF-IDF vectorization and a Logistic Regression model to predict emotions such as joy, sadness, anger, fear, etc., and is deployed as a web application using Flask on Render.

Project Overview:

1.The system follows a complete machine learning pipeline:
2.Text preprocessing (cleaning, tokenization, stemming, stopword removal)
3.Feature extraction using TF-IDF
4.Model training using Logistic Regression
5.Label encoding for emotion classes
6.Deployment using Flask and Gunicorn
7.Users can enter any text through the web interface, and the model predicts the corresponding emotion in real-time.

Files and Structure:

app.py – Flask backend handling routing and predictions
templates/index.html – Frontend UI
Emotion_Classification.ipynb – Model training and preprocessing workflow
train.txt, test.txt, val.txt – Dataset files

Requirements:

The project dependencies are listed in requirements.txt, including:
1.Flask (web framework)
2.Pandas, NumPy (data handling)
3.Scikit-learn (ML model and TF-IDF)
4.NLTK (text preprocessing)
5.Gunicorn (production server)
These are pinned to stable versions to ensure compatibility and smooth deployment.

Runtime & Python Version:

The runtime.txt file specifies:
python-3.10.13
This ensures compatibility with machine learning libraries like NumPy, Pandas, and Scikit-learn, avoiding build and dependency errors.

Deployment:
The application is deployed on Render, using a Procfile:
web: gunicorn app:app
Note: On the free tier, the app may experience a short delay due to cold starts.

🔗 *Live Demo:* [Emotion Classification Web App](https://emotion-classification-project-14.onrender.com)

Conclusion:

This project demonstrates practical skills in NLP, machine learning, and web deployment, making it suitable for real-world applications and portfolio presentation.
