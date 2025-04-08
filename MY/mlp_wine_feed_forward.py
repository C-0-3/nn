import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Load the Wine dataset from sklearn
wine_data = datasets.load_wine()

# Features (X) and Labels (y)
X = wine_data.data
y = wine_data.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train MLP on the Wine dataset and display results
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("Training MLP on Wine dataset")
    
    # Define MLP model (single hidden layer with 50 neurons)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    
    # Train the model
    mlp.fit(X_train, y_train)
    
    # Predicting the output on the test set
    y_pred = mlp.predict(X_test)
    
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Wine dataset: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for Wine dataset:\n", cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
    plt.title("Confusion Matrix for Wine dataset")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Train and evaluate the model
train_and_evaluate(X_train, X_test, y_train, y_test)
