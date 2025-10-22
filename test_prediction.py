from src.predict import predict_news

print("📰 Fake News Detector - Local Test\n")

while True:
    text = input("Enter a news headline or article (or type 'exit' to quit):\n> ")
    if text.lower() == 'exit':
        break
    result = predict_news(text)
    print(f"\nPrediction: {result}\n")
