# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Load dataset
data = pd.read_csv("Emotion_final.csv")  

# Train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Text'])
y = data['Emotion']
classifier = LinearSVC()
classifier.fit(X, y)

# Save trained model and vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(classifier, "classifier.pkl")

# Function to predict emotion
def predict_emotion(text):
    # Load vectorizer and classifier
    vectorizer = joblib.load("vectorizer.pkl")
    classifier = joblib.load("classifier.pkl")

    # Preprocess input text and predict emotion
    text_vec = vectorizer.transform([text])
    prediction = classifier.predict(text_vec)

    return prediction[0]

# Example usage
# input_text = "Today was one of those days that remind me to appreciate the little things in life. The morning greeted me with a gentle breeze and the promise of a beautiful day ahead. As I sipped my morning coffee, I couldn't help but feel a sense of gratitude for the simple pleasures that each day bring"
# predicted_emotion = predict_emotion(input_text)
# print("Predicted Emotion:", predicted_emotion)

