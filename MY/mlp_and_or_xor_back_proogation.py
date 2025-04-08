import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# AND dataset
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # AND truth table

# OR dataset
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])  # OR truth table

# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR truth table

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# MLP Class with Backpropagation
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
        
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output
    
    def backward(self, X, y, output):
        # Calculate the error
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        # Backpropagate to hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
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

# Train and evaluate the AND dataset
mlp_and = MLP(input_size=2, hidden_size=2, output_size=1)
mlp_and.train(X_and, y_and.reshape(-1, 1), epochs=10000)

# Prediction
y_pred_and = mlp_and.predict(X_and)
y_pred_and = np.round(y_pred_and)

# Calculate Accuracy
accuracy_and = accuracy_score(y_and, y_pred_and)
print(f"Accuracy on AND dataset: {accuracy_and * 100:.2f}%")

# Confusion Matrix
cm_and = confusion_matrix(y_and, y_pred_and)
print(f"Confusion Matrix for AND dataset:\n", cm_and)

# Plot Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(cm_and, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix for AND dataset")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Train and evaluate the OR dataset
mlp_or = MLP(input_size=2, hidden_size=2, output_size=1)
mlp_or.train(X_or, y_or.reshape(-1, 1), epochs=10000)

# Prediction
y_pred_or = mlp_or.predict(X_or)
y_pred_or = np.round(y_pred_or)

# Calculate Accuracy
accuracy_or = accuracy_score(y_or, y_pred_or)
print(f"Accuracy on OR dataset: {accuracy_or * 100:.2f}%")

# Confusion Matrix
cm_or = confusion_matrix(y_or, y_pred_or)
print(f"Confusion Matrix for OR dataset:\n", cm_or)

# Plot Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(cm_or, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix for OR dataset")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Train and evaluate the XOR dataset
mlp_xor = MLP(input_size=2, hidden_size=2, output_size=1)
mlp_xor.train(X_xor, y_xor.reshape(-1, 1), epochs=10000)

# Prediction
y_pred_xor = mlp_xor.predict(X_xor)
y_pred_xor = np.round(y_pred_xor)

# Calculate Accuracy
accuracy_xor = accuracy_score(y_xor, y_pred_xor)
print(f"Accuracy on XOR dataset: {accuracy_xor * 100:.2f}%")

# Confusion Matrix
cm_xor = confusion_matrix(y_xor, y_pred_xor)
print(f"Confusion Matrix for XOR dataset:\n", cm_xor)

# Plot Confusion Matrix
plt.figure(figsize=(5, 5))
sns.heatmap(cm_xor, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix for XOR dataset")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
