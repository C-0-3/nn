import tensorflow as tf
from tensorflow.keras import layers, models
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Show first 9 training images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap='gray')
    plt.axis('off')
plt.tight_layout(); plt.show()

# Preprocess
train_X = train_X.reshape(-1, 28, 28, 1).astype('float32') / 255
test_X = test_X.reshape(-1, 28, 28, 1).astype('float32') / 255

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_X, train_y, epochs=5, validation_data=(test_X, test_y))

# Evaluate
test_loss, test_acc = model.evaluate(test_X, test_y)
print(f"Test Accuracy: {test_acc}")

# Plot training vs test accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curve')
plt.show()

# Predict first 5 test images and show
predictions = model.predict(test_X[:5])
for i in range(5):
    plt.imshow(test_X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predictions[i].argmax()}, True: {test_y[i]}")
    plt.axis('off'); plt.show()

# Save model
model.save('digit_recognition_model.h5')
