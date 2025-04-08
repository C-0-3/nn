import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_wine
import seaborn as sns

# ReLU activation function and derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax activation function for output layer
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Softmax derivative for output layer
def softmax_derivative(y):
    # For simplicity, we'll use the derivative of the cross-entropy loss with softmax
    # Since we're not using it directly in backprop, we'll focus on the cross-entropy loss instead
    pass

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred))

# Cross-entropy derivative
def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

# MLP Class with Backpropagation
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
        self.hidden_output = relu(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        # Using softmax for output layer
        self.output = softmax(self.output_input)
        return self.output
    
    def backward(self, X, y, output):
        # Cross-entropy derivative
        output_error = cross_entropy_derivative(y, output)
        
        # Output delta
        output_delta = output_error
        
        # Hidden error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        
        # Hidden delta
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

# Load Wine dataset
wine_data = load_wine()
X_wine = wine_data.data
y_wine = wine_data.target

# One-hot encoding for labels
y_wine_onehot = np.eye(3)[y_wine]

# Split dataset into training and testing sets
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine_onehot, test_size=0.3, random_state=42)

# Train and evaluate the Wine dataset
mlp_wine = MLP(input_size=X_train_wine.shape[1], hidden_size=50, output_size=3)
mlp_wine.train(X_train_wine, y_train_wine, epochs=10000)

# Prediction
y_pred_wine = mlp_wine.predict(X_test_wine)
y_pred_wine_class = np.argmax(y_pred_wine, axis=1)

# Calculate Accuracy
accuracy_wine = accuracy_score(np.argmax(y_test_wine, axis=1), y_pred_wine_class)
print(f"Accuracy on Wine dataset: {accuracy_wine * 100:.2f}%")

# Confusion Matrix
cm_wine = confusion_matrix(np.argmax(y_test_wine, axis=1), y_pred_wine_class)
print(f"Confusion Matrix for Wine dataset:\n", cm_wine)

# Plot Confusion Matrix
plt.figure(figsize=(7, 7))
sns.heatmap(cm_wine, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Wine dataset")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
