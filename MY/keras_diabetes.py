# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Load Diabetes dataset from sklearn
from sklearn.datasets import load_diabetes

# Uncomment to load CSV data
# df = pd.read_csv('diabetes_data.csv')
# X = df.drop('target', axis=1)
# y = df['target']

data = load_diabetes()
X = data.data
y = data.target
y = (y > np.median(y)).astype(int)  # Binarize target for classification

# Preprocess the data (train-test split and scaling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Keras model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification for Diabetes

# Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluate model and predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(2), ['No Diabetes', 'Diabetes'])
plt.yticks(np.arange(2), ['No Diabetes', 'Diabetes'])
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Show individual predictions for the first 5 test samples
print("\nPredictions for the first 5 test samples:")
for i in range(5):
    print(f"Sample {i+1}: Actual = {'Diabetes' if y_test[i] == 1 else 'No Diabetes'}, Predicted = {'Diabetes' if y_pred[i] == 1 else 'No Diabetes'}")
