import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Function to load CSV data
def load_csv_data(file_path):
    # Load the dataset from a CSV file
    data = pd.read_csv(file_path)
    return data

# Function to train MLP on a dataset and display results
def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name):
    print(f"Training MLP on {dataset_name} dataset")
    
    # Define MLP model (single hidden layer with 50 neurons)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    
    # Train the model
    mlp.fit(X_train, y_train)
    
    # Predicting the output on the test set
    y_pred = mlp.predict(X_test)
    
    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on {dataset_name} dataset: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for {dataset_name} dataset:\n", cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Loading the Diabetes dataset from CSV file
diabetes_file = 'path_to_your_diabetes.csv'  # Specify the correct path to your diabetes dataset
diabetes_data = load_csv_data(diabetes_file)

# Splitting Diabetes data into features and target
X_diabetes = diabetes_data.drop(columns=['Outcome'])  # Assuming 'Outcome' is the target
y_diabetes = diabetes_data['Outcome']

# Splitting Diabetes dataset into training and test sets
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes, test_size=0.3, random_state=42)

# Training and evaluating the Diabetes dataset
train_and_evaluate(X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes, "Diabetes")

# Loading the Wine dataset from CSV file
wine_file = 'path_to_your_wine.csv'  # Specify the correct path to your wine dataset
wine_data = load_csv_data(wine_file)

# Splitting Wine data into features and target
X_wine = wine_data.drop(columns=['class'])  # Assuming 'class' is the target
y_wine = wine_data['class']

# Splitting Wine dataset into training and test sets
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)

# Training and evaluating the Wine dataset
train_and_evaluate(X_train_wine, X_test_wine, y_train_wine, y_test_wine, "Wine")
