import os
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_PATH = os.path.join("data", "mnist.npz")

# Function to load MNIST data (the same file Keras uses, fetched once and
# cached, so TensorFlow isn't needed just to read it)
def load_mnist():
    if not os.path.exists(MNIST_PATH):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL, MNIST_PATH)
    with np.load(MNIST_PATH) as f:
        x_train, y_train, x_test, y_test = f["x_train"], f["y_train"], f["x_test"], f["y_test"]
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0
    y_train_binary = (y_train == 6).astype(int)
    y_test_binary = (y_test == 6).astype(int)
    return x_train, y_train_binary, x_test, y_test_binary, y_test

# Function to perform PCA: fit on the training data only, then project both
# sets onto those same components
def perform_pca(train_data, test_data, desired_variance_ratio):
    pca = PCA(n_components=desired_variance_ratio, svd_solver='full')
    train_transformed = pca.fit_transform(train_data)
    test_transformed = pca.transform(test_data)
    return train_transformed, test_transformed, pca.n_components_

# Function to train and evaluate logistic regression model
def train_and_evaluate_logistic_regression(x_train, y_train, x_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    classification_rep = classification_report(y_test, y_test_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    misclassified_indices = np.where(y_test != y_test_pred)[0]
    misclassified_samples = x_test[misclassified_indices]
    return train_accuracy, test_accuracy, classification_rep, conf_matrix, misclassified_indices, misclassified_samples

# Load MNIST data
x_train, y_train, x_test, y_test, test_digits = load_mnist()

# Train and evaluate logistic regression on MNIST data
mnist_accuracy_train, mnist_accuracy_test, mnist_classification_rep, mnist_conf_matrix, mnist_misclassified_indices, mnist_misclassified_samples = train_and_evaluate_logistic_regression(x_train, y_train, x_test, y_test)

# Print MNIST results
print("MNIST Results (is it a 6?):")
print(f"Number of features (pixels): {x_train.shape[1]}")
print(f"Training Accuracy: {mnist_accuracy_train:.2f}")
print(f"Testing Accuracy: {mnist_accuracy_test:.2f}")
print("Classification Report:\n", mnist_classification_rep)
print("Confusion Matrix:\n", mnist_conf_matrix)
print(f"Number of Misclassified Samples: {len(mnist_misclassified_indices)}")

# Which actual digits were misclassified (0-9, not the 6-or-not label)
digits, counts = np.unique(test_digits[mnist_misclassified_indices], return_counts=True)
print("Misclassified digits (digit: count):", dict(zip(digits.tolist(), counts.tolist())))

# Sample text data for language prediction
texts = [
    "Bonjour, comment ça va?", "Hello, how are you?", "Hola, ¿cómo estás?", "Guten Tag, wie geht es Ihnen?",
    "Je suis fatigué.", "I am tired.", "Estoy cansado.", "Ich bin müde.",
    "Le soleil brille.", "The sun is shining.", "El sol brilla.", "Die Sonne scheint."
]
labels = ["French", "English", "Spanish", "German", "French", "English", "Spanish", "German",
          "French", "English", "Spanish", "German"]

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train and evaluate logistic regression on text data
text_accuracy_train, text_accuracy_test, text_classification_rep, _, text_misclassified_indices, text_misclassified_samples = train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test)

# Create and train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Get the predicted labels
predicted_labels = model.predict(X_test)

# Print text data results
print("\nLanguage Prediction Results:")
print(f"Number of TF-IDF features (vocabulary size): {X_train.shape[1]}")
print(f"Training Accuracy: {text_accuracy_train:.2f}")
print(f"Testing Accuracy: {text_accuracy_test:.2f}")
print("Classification Report:\n", text_classification_rep)
print(f"Number of Misclassified Samples: {len(text_misclassified_indices)}")

# Extract and print the misclassified labels (indices are into the test split)
misclassified_labels = np.array(y_test)[np.array(text_misclassified_indices)]
print("Misclassified Labels:", misclassified_labels)

# Print the predicted labels for all test samples
print("Predicted Labels:", predicted_labels)


# Same MNIST split as above, so the two results compare directly
# (this used to re-download MNIST from OpenML and fit a separate PCA on
# the test set, which put train and test on different axes). Reloaded from
# the cached file because the language section reused y_train / y_test.
x_train, y_train, x_test, y_test, test_digits = load_mnist()
X_train_pca, X_test_pca, num_components_preserved = perform_pca(x_train, x_test, 0.88)

# Train and evaluate logistic regression on PCA-transformed data
pca_accuracy_train, pca_accuracy_test, pca_classification_rep, pca_conf_matrix, pca_misclassified_indices, pca_misclassified_samples = train_and_evaluate_logistic_regression(X_train_pca, y_train, X_test_pca, y_test)

# Print PCA data results
print("\nPCA Results:")
print(f"Number of Principal Components Preserved: {num_components_preserved}")
print(f"Training Accuracy: {pca_accuracy_train:.2f}")
print(f"Testing Accuracy: {pca_accuracy_test:.2f}")
print("Classification Report:\n", pca_classification_rep)
print("Confusion Matrix:\n", pca_conf_matrix)
print(f"Number of Misclassified Samples: {len(pca_misclassified_indices)}")

# Which actual digits were misclassified after PCA
digits, counts = np.unique(test_digits[pca_misclassified_indices], return_counts=True)
print("Misclassified digits (digit: count):", dict(zip(digits.tolist(), counts.tolist())))
