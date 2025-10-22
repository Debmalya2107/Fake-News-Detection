# 📰 Fake News Detection

## 📘 Overview
This project aims to detect whether a news article is **real or fake** using Natural Language Processing (NLP) and Machine Learning.  
The system preprocesses textual data, vectorizes it using a trained vectorizer, and predicts authenticity with a trained model.

---

## 🚀 Features
- Text preprocessing and cleaning  
- Model training and evaluation  
- Real-time prediction using saved ML models  
- Easy-to-use structure for retraining or testing new data  

---

## 🧮 Model Details

Algorithm: Logistic Regression / Naive Bayes / SVM (depending on training)

Vectorizer: TF-IDF or Count Vectorizer

The model and vectorizer are pre-trained and stored under the models/ directory.

You can retrain the model using src/model_training.py.
