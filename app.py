from flask import Flask, render_template, request, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from langdetect import detect
from SVM import SVM
from sentiment_analysis import SentimentIntensityAnalyzer, get_sentiment_features
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import load

app = Flask(__name__)

app.secret_key = 'your_secret_key'  # Set a secret key for security purposes

# Load the saved model and its performance metrics
model_data = load('svm_model_and_performance_BALANCE_0-1.joblib')
svm_classifier = model_data['model']
model_metrics = model_data['performance_metrics']

# Load the dataset
dataset = pd.read_csv('FinalDataset.csv')  # Replace with your dataset file

# Extract features and labels
texts = dataset['tweet_text'].str.lower().values  # Replace with your text column name
labels = dataset['cyberbullying_type'].values  # Replace with your label column name

# Convert labels to numeric format
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into 60% train and 40% remaining
text_train, text_temp, labels_train, labels_temp = train_test_split(texts, labels_encoded, test_size=0.4, random_state=42)
# Further split the 40% into 30% test and 10% validation (which is 75%/25% of the 40%)
text_test, text_val, labels_test, labels_val = train_test_split(text_temp, labels_temp, test_size=0.25, random_state=42)

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(text_train)
X_test_tfidf = tfidf_vectorizer.transform(text_test)

# Initialize VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Get sentiment features
sentiment_features_train = get_sentiment_features(text_train, sia)
sentiment_features_test = get_sentiment_features(text_test, sia)

# Combine TF-IDF features with sentiment features
X_train_combined = np.concatenate([X_train_tfidf.toarray(), sentiment_features_train], axis=1)
X_test_combined = np.concatenate([X_test_tfidf.toarray(), sentiment_features_test], axis=1)

# Convert labels for SVM
labels_train_svm = np.where(labels_train == 0, -1, 1)
labels_test_svm = np.where(labels_test == 0, -1, 1)

# Create an instance of the SVM classifier and train it
# svm_classifier = SVM()  # Commenting out model initialization
# svm_classifier.fit(X_train_combined, labels_train_svm)  # Commenting out model training

# Calculate classification report
predictions = svm_classifier.predict(X_test_combined)
# Convert predictions back to the original label encoding
predictions = np.where(predictions == -1, 0, 1)

report = classification_report(labels_test, predictions, output_dict=True)
accuracy = accuracy_score(labels_test, predictions)
accuracy = report['accuracy']

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the classification result
@app.route('/classify', methods=['POST'])
def classify():
    sample_text = request.form['text_area'].lower()
    
    # Detect language
    detected_language = detect(sample_text)
    
    hashtags = re.findall(r"#(\w+)", sample_text)

    
    # Vectorize the sample text using the same TfidfVectorizer
    sample_text_vectorized = tfidf_vectorizer.transform([sample_text])
    
    # Get sentiment features for the sample text
    sample_sentiment_features = get_sentiment_features([sample_text], sia)
    
    # Combine TF-IDF features with sentiment features for the sample text
    sample_text_combined = np.concatenate([sample_text_vectorized.toarray(), sample_sentiment_features], axis=1)
    
    # Ensure the combined feature set matches the training data's feature dimension
    if sample_text_combined.shape[1] != X_train_combined.shape[1]:
        raise ValueError("The feature set of the sample text does not match the training data.")
    
    # Predict using the trained SVM model
    prediction = svm_classifier.predict(sample_text_combined)
    
    # Convert prediction to original label
    predicted_label = label_encoder.inverse_transform(np.where(prediction == -1, 0, 1))[0]

    # Get the sentiment score
    sentiment_score = sia.polarity_scores(sample_text)
    compound_score = sentiment_score['compound']
    
    session['classification_result'] = predicted_label
    session['sentiment_score'] = sentiment_score
    session['detected_language'] = detected_language
    session['hashtags'] = hashtags
    session['input_text'] = sample_text

    
    return render_template('index.html', result=predicted_label)

# Define a route for displaying performance metrics

@app.route('/metrics')
def metrics():
    
    classification_result = session.get('classification_result', None)
    sentiment_score = session.get('sentiment_score', None)
    detected_language = session.get('detected_language', None)
    hashtags = session.get('hashtags', None)
    input_text = session.get('input_text', None)
    return render_template('metrics.html', result=classification_result, sentiment_score=sentiment_score, detected_language=detected_language, hashtags=hashtags, input_text=input_text)
    
if __name__ == '__main__':
    app.run(debug=True)