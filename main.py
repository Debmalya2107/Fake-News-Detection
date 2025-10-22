from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_models
from src.predict import predict_news

if __name__ == "__main__":
    print("📥 Loading and preprocessing data...")

    files_labels_cols = [
        ("data/True.csv", 1, None),
        ("data/Fake.csv", 0, None),
        ("data/social_media_misinformation.csv", 0, None),
        ("data/fake_news_2024.csv", 1, None)
    ]

    news = load_and_preprocess_data(files_labels_cols)
    print(f"✅ Total articles loaded: {len(news)}")

    print("🧠 Training models...")
    train_models(news)

    print("🔍 Testing on a new article...")
    sample_text = "The government has announced new environmental policies."
    prediction = predict_news(sample_text)
    print("Prediction:", prediction)
