import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# ReLU activation function and derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax activation function
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# MLP Class with Backpropagation using Softmax
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)  # Using ReLU for hidden layers
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = softmax(self.output_input)  # Using Softmax for multi-class output
        return self.output
    
    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error  # No softmax derivative as we are using a direct error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * relu_derivative(self.hidden_output)
        
        self.weights_input_hidden += X.T.dot(hidden_delta)
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)
        
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
        
    def predict(self, X):
        return self.forward(X)

# Load CSV data
def load_csv_data(file_path):
    return pd.read_csv(file_path)

# Main function to handle both datasets
def main():
    # Load Wine dataset
    wine_data = load_csv_data('wine.csv')  # Update the path
    X_wine = wine_data.drop(columns=['class'])
    y_wine = wine_data['class']

    # Load Diabetes dataset
    diabetes_data = load_csv_data('diabetes.csv')  # Update the path
    X_diabetes = diabetes_data.drop(columns=['Outcome'])
    y_diabetes = diabetes_data['Outcome']

    # Preprocess and split data
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)
    X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)

    # Convert labels to one-hot for both datasets
    y_train_wine_onehot = np.zeros((len(y_train_wine), 3))
    y_train_wine_onehot[np.arange(len(y_train_wine)), y_train_wine - 1] = 1  # Adjust for Wine dataset labels starting from 1

    y_train_diabetes_onehot = np.zeros((len(y_train_diabetes), 2))
    y_train_diabetes_onehot[np.arange(len(y_train_diabetes)), y_train_diabetes] = 1

    # Train and evaluate Wine dataset
    mlp_wine = MLP(input_size=X_train_wine.shape[1], hidden_size=50, output_size=3)
    mlp_wine.train(X_train_wine.values, y_train_wine_onehot, epochs=10000)

    # Prediction for Wine dataset
    y_pred_wine = mlp_wine.predict(X_test_wine.values)
    y_pred_wine = np.argmax(y_pred_wine, axis=1) + 1  # Convert probabilities to class labels

    # Calculate Accuracy for Wine dataset
    accuracy_wine = accuracy_score(y_test_wine, y_pred_wine)
    print(f"Accuracy on Wine dataset: {accuracy_wine * 100:.2f}%")

    # Confusion Matrix for Wine dataset
    cm_wine = confusion_matrix(y_test_wine, y_pred_wine)
    print(f"Confusion Matrix for Wine dataset:\n", cm_wine)

    # Plot Confusion Matrix for Wine dataset
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm_wine, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_wine), yticklabels=np.unique(y_wine))
    plt.title("Confusion Matrix for Wine dataset")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Train and evaluate Diabetes dataset
    mlp_diabetes = MLP(input_size=X_train_diabetes.shape[1], hidden_size=50, output_size=2)
    mlp_diabetes.train(X_train_diabetes.values, y_train_diabetes_onehot, epochs=10000)

    # Prediction for Diabetes dataset
    y_pred_diabetes = mlp_diabetes.predict(X_test_diabetes.values)
    y_pred_diabetes = np.argmax(y_pred_diabetes, axis=1)  # Convert probabilities to class labels

    # Calculate Accuracy for Diabetes dataset
    accuracy_diabetes = accuracy_score(y_test_diabetes, y_pred_diabetes)
    print(f"Accuracy on Diabetes dataset: {accuracy_diabetes * 100:.2f}%")

    # Confusion Matrix for Diabetes dataset
    cm_diabetes = confusion_matrix(y_test_diabetes, y_pred_diabetes)
    print(f"Confusion Matrix for Diabetes dataset:\n", cm_diabetes)

    # Plot Confusion Matrix for Diabetes dataset
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm_diabetes, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title("Confusion Matrix for Diabetes dataset")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()
