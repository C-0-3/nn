# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing: Label encode the target variable (classes)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the Keras neural network model
model = Sequential()

# Input layer: 4 input features (sepal_length, sepal_width, petal_length, petal_width)
model.add(Dense(10, input_dim=4, activation='relu'))

# Hidden layer: 10 neurons
model.add(Dense(10, activation='relu'))

# Output layer: 3 classes (since Iris has 3 types of species)
model.add(Dense(3, activation='softmax'))

# Compile the model: Use categorical crossentropy for multi-class classification and adam optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate the accuracy using sklearn
accuracy_sklearn = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy using sklearn: {accuracy_sklearn * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Visualize the training history (optional)
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Example prediction on a random test sample
sample_idx = 0  # You can change the index for different predictions
sample_input = X_test[sample_idx].reshape(1, -1)
sample_prediction = model.predict(sample_input)
predicted_class = np.argmax(sample_prediction)
class_labels = ['Setosa', 'Versicolor', 'Virginica']
print(f"Predicted class for sample {sample_idx}: {class_labels[predicted_class]}")

# Print probabilities for each class
print(f"Probabilities for each class: {sample_prediction[0]}")

# Visualize class probabilities
plt.figure(figsize=(8, 6))
plt.bar(class_labels, sample_prediction[0])
plt.title('Class Probabilities for Sample')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.show()
