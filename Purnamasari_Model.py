import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler  # For scaling features
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
import math
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from All_In_One_Bullytech_Model import get_sentiment_features

import nltk
nltk.download('punkt')


# Function definitions (preprocess_text as you defined it)
def preprocess_text(text):
    # Tokenizing
    words = word_tokenize(text)

    # Filtering (Removing stopwords and punctuations)
    words = [word for word in words if word.isalpha()]  # Removes punctuation
    stop_words = set(stopwords.words('english'))  # Using English stop words
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

# Function to calculate entropy
def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum([p * math.log2(p) for p in probabilities])
    return entropy

# Function to calculate information gain
def information_gain(X, y):
    total_entropy = calculate_entropy(y)
    n_docs = X.shape[0]
    ig_scores = []

    for i in range(X.shape[1]):
        X_feature = X[:, i]
        y_feature_present = y[X_feature.nonzero()[0]]
        y_feature_absent = y[np.where(X_feature.toarray().ravel() == 0)]
        entropy_present = calculate_entropy(y_feature_present)
        entropy_absent = calculate_entropy(y_feature_absent)
        p_feature_present = len(y_feature_present) / n_docs
        p_feature_absent = len(y_feature_absent) / n_docs
        ig = total_entropy - p_feature_present * entropy_present - p_feature_absent * entropy_absent
        ig_scores.append(ig)

    return ig_scores

# Read dataset and preprocess texts (as you have already defined)
dataset = pd.read_csv('Purnamasari_300_Dataset.csv')  # Replace with your dataset file

texts = dataset['tweet_text'].str.lower().values  # Replace with your text column name
labels = dataset['cyberbullying_type'].values  # Replace with your label column name

processed_texts = [preprocess_text(text) for text in texts]

# Term Weighting with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_texts)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Calculate Information Gain
ig_scores = information_gain(X_train, y_train)

# Select features based on Information Gain
# Example: Select top N features
N = 1000  # You can adjust this value
top_features = np.argsort(ig_scores)[-N:]

# Apply feature selection to X_train and X_test
X_train_selected = X_train[:, top_features]
X_test_selected = X_test[:, top_features]

C = 1        # Regularization parameter
gamma = 0.001  # Kernel coefficient
max_iter = 20  # Maximum number of iterations

# Define the SVM classifier with a polynomial kernel
svm_classifier = SVC(kernel='poly', C=C, gamma=gamma, max_iter=max_iter)

# Train the classifier on the training data
svm_classifier.fit(X_train_selected, y_train)

# Predict on the test data
y_pred = svm_classifier.predict(X_test_selected)

# Evaluate the classifier
print(classification_report(y_test, y_pred))
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')


# Print accuracy, precision, recall, and F1-score
print("Accuracy:", accuracy)
print("Precision (weighted average):", precision)
print("Recall (weighted average):", recall)
print("F1-score (weighted average):", f1_score)