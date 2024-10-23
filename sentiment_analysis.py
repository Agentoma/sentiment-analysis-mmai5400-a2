import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the training data
train_data = pd.read_csv('train.csv')

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer and transform the training data
X_train = vectorizer.fit_transform(train_data['Review'])

# Extract the labels
y_train = train_data['Sentiment']

# Check a sample of the transformed TF-IDF feature matrix to verify the process
import numpy as np

# Get the TF-IDF feature names and inspect a sample of transformed data
sample_feature_names = vectorizer.get_feature_names_out()[:10]
print("Sample feature names:", sample_feature_names)

# Shape of Transformed Data
transformed_data_shape = X_train.shape
print("Shape of transformed data:", transformed_data_shape)

# Convert to dense to inspect actual values for a sample row (first 10 features)
sample_dense = X_train[0].toarray()[0][:10]
print("Sample dense row:", sample_dense)

# Calculate sparsity directly from sparse matrix
sparsity = 1.0 - (X_train.nnz / X_train.shape[0] / X_train.shape[1])
print("Sparsity of the matrix:", sparsity)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

def evaluate(filename):
    # Load the data
    data = pd.read_csv(filename)
    
    # Transform the reviews using the fitted vectorizer
    X = vectorizer.transform(data['Review'])
    y_true = data['Sentiment']
    
    # Predict the sentiments
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Calculate macro-averaged F1-score
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"Average F1 score: {f1_macro:.4f}")
    
    # Calculate class-wise F1-scores
    f1_classwise = f1_score(y_true, y_pred, average=None)
    print("Class-wise F1 scores:")
    for i, label in enumerate(['negative', 'neutral', 'positive']):
        print(f"{label}: {f1_classwise[i]:.4f}")
    
    # Calculate and display the normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'neutral', 'positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.show()

# Evaluate the model on the validation set
evaluate('valid.csv')
