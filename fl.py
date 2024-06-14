from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
vectorizer = joblib.load("vectorizer.pkl")
classifier = joblib.load("classifier.pkl")

# Function to predict emotion
def predict_emotion(text):
    text_vec = vectorizer.transform([text])
    prediction = classifier.predict(text_vec)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        journal_entry = request.form['journal_entry']
        predicted_emotion = predict_emotion(journal_entry)
        return render_template('index.html', journal_entry=journal_entry, predicted_emotion=predicted_emotion)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
