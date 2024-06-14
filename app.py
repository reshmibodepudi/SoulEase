from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import random

app = Flask(__name__)

# Load sentiment analysis pipeline with specific model and revision
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Generate affirmation based on mood
def generate_affirmation(mood):
    if mood == "POSITIVE":
        affirmations = ["You're doing great! Keep it up!", "You're on the right track!", "You inspire others with your positivity!"]
        return random.choice(affirmations)
    elif mood == "NEGATIVE":
        affirmations = ["Tough times don't last, tough people do. You've got this!", "Every challenge is an opportunity for growth.", "You're stronger than you think!"]
        return random.choice(affirmations)
    else:
        return "Keep going, you're doing great!"

@app.route('/')
def index():
    return render_template('affir.html')

@app.route('/process', methods=['POST'])
def process():
    journal_entry = request.form['journal_entry']
    print("Received journal entry:", journal_entry)  # Print the journal entry to console
    
    # Perform sentiment analysis on journal entry
    sentiment_result = sentiment_pipeline(journal_entry)
    mood = sentiment_result[0]['label']

    # Generate affirmation based on mood
    affirmation = generate_affirmation(mood)

    return jsonify({'affirmation': affirmation})

if __name__ == '__main__':
    app.run(debug=True)
