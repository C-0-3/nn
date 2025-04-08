import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Defining the datasets: AND, OR, XOR
# Each dataset will have input values as pairs of binary values (0, 1)
# and a corresponding target value based on the logic operation.

# AND Dataset
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # AND truth table

# OR Dataset
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])  # OR truth table

# XOR Dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR truth table

# Function to train MLP on the dataset and display results
def train_and_evaluate(X, y, dataset_name):
    print(f"Training MLP on {dataset_name} dataset")
    
    # Define MLP model (single hidden layer with 2 neurons)
    mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=42)
    
    # Train the model
    mlp.fit(X, y)
    
    # Predicting the output
    y_pred = mlp.predict(X)
    
    # Calculating accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy on {dataset_name} dataset: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix for {dataset_name} dataset:\n", cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Evaluate for AND, OR, and XOR datasets
train_and_evaluate(X_and, y_and, "AND")
train_and_evaluate(X_or, y_or, "OR")
train_and_evaluate(X_xor, y_xor, "XOR")
