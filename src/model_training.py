from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def train_models(news_df):
    X = news_df['clean_text']
    y = news_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train models
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vec, y_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    # Evaluate
    def evaluate_model(model, name):
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} Performance:")
        print("Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        return acc

    lr_acc = evaluate_model(lr_model, "Logistic Regression")
    nb_acc = evaluate_model(nb_model, "Naive Bayes")

    # Select best model
    best_model = lr_model if lr_acc >= nb_acc else nb_model
    best_model_name = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"
    print(f"\n✅ Best model selected: {best_model_name} with accuracy {max(lr_acc, nb_acc):.4f}")

    # Save model/vectorizer
    os.makedirs('models', exist_ok=True)
    with open('models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✅ Model and vectorizer saved successfully.")
