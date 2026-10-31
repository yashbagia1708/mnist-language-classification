import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to load MNIST data
def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0
    y_train_binary = (y_train == 6).astype(int)
    y_test_binary = (y_test == 6).astype(int)
    return x_train, y_train_binary, x_test, y_test_binary

# Function to perform PCA and return transformed data
def perform_pca(data, desired_variance_ratio):
    pca = PCA(n_components=desired_variance_ratio, svd_solver='full')
    transformed_data = pca.fit_transform(data)
    num_components_preserved = pca.n_components_
    return transformed_data, num_components_preserved

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
x_train, y_train, x_test, y_test = load_mnist()

# Train and evaluate logistic regression on MNIST data
mnist_accuracy_train, mnist_accuracy_test, mnist_classification_rep, mnist_conf_matrix, mnist_misclassified_indices, mnist_misclassified_samples = train_and_evaluate_logistic_regression(x_train, y_train, x_test, y_test)

# Print MNIST results
print("MNIST Results:")
print(f"Number of Principal Components Preserved: {x_train.shape[1]}")
print(f"Training Accuracy: {mnist_accuracy_train:.2f}")
print(f"Testing Accuracy: {mnist_accuracy_test:.2f}")
print("Classification Report:\n", mnist_classification_rep)
print("Confusion Matrix:\n", mnist_conf_matrix)
print(f"Number of Misclassified Samples: {len(mnist_misclassified_indices)}")

# Print the misclassified digits
misclassified_digits = y_test[mnist_misclassified_indices]
print("Misclassified Digits:", misclassified_digits)

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
print(f"Number of Principal Components Preserved: {X_train.shape[1]}")
print(f"Training Accuracy: {text_accuracy_train:.2f}")
print(f"Testing Accuracy: {text_accuracy_test:.2f}")
print("Classification Report:\n", text_classification_rep)
print(f"Number of Misclassified Samples: {len(text_misclassified_indices)}")

# Extract and print the misclassified labels
misclassified_labels = np.array(labels)[np.array(text_misclassified_indices)]
print("Misclassified Labels:", misclassified_labels)

# Print the predicted labels for all test samples
print("Predicted Labels:", predicted_labels)


# Load the MNIST dataset
mnist = fetch_openml("mnist_784", parser='auto')
X, y = mnist.data, mnist.target
y = np.where(y == '6', 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PCA and train/evaluate logistic regression
X_train_pca, num_components_preserved = perform_pca(X_train, 0.88)
X_test_pca = PCA(n_components=num_components_preserved).fit_transform(X_test)

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

# Print the misclassified digits
misclassified_digits = y_test[mnist_misclassified_indices]
print("Misclassified Digits:", misclassified_digits)
