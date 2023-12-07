import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler  # For scaling features
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split, KFold

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Load dataset
dataset = pd.read_csv('FinalDataset.csv')  # Replace with the path to your dataset

# Apply preprocessing to the text
dataset['processed_text'] = dataset['tweet_text'].apply(preprocess_text)  # Assuming 'tweet_text' is the column name

# TF-IDF Vectorization with unigram
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=5000)
X = vectorizer.fit_transform(dataset['processed_text'])
y = dataset['cyberbullying_type']  # Assuming 'cyberbullying_type' is the label column

# Scaling the features (if necessary)
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing set (80%-20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

# Define the KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the model with RBF kernel
model = SVC(kernel='rbf')

# Perform k-fold cross-validation
for train_index, val_index in kf.split(X_train_full):
    X_train, X_val = X_train_full[train_index], X_train_full[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    predictions = model.predict(X_val)
    print(f"Validation Accuracy for RBF kernel: {accuracy_score(y_val, predictions)}")
    print(f"Validation Classification Report for RBF kernel:\n{classification_report(y_val, predictions)}")

# Final evaluation on the test set
print("\nFinal Evaluation on Test Set")
best_model = SVC(kernel='rbf')
best_model.fit(X_train_full, y_train_full)
final_predictions = best_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, final_predictions)}")
print(f"Test Classification Report:\n{classification_report(y_test, final_predictions)}")
