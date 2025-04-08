import numpy as np
import matplotlib.pyplot as plt

# Activation function: Threshold
def threshold_activation(x):
    return 1 if x >= 0 else 0

# Perceptron class for threshold activation function
class PerceptronThreshold:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size + 1)  # weights + bias
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return threshold_activation(summation)

    def train(self, inputs, targets, epochs=100):
        for epoch in range(epochs):
            for input_data, target in zip(inputs, targets):
                prediction = self.predict(input_data)
                error = target - prediction
                self.weights[1:] += self.learning_rate * error * input_data
                self.weights[0] += self.learning_rate * error

    def evaluate(self, inputs, targets):
        predictions = [self.predict(input_data) for input_data in inputs]
        accuracy = np.mean(np.array(predictions) == np.array(targets))
        return accuracy, predictions


# Datasets: AND, OR, XOR
datasets = {
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    "OR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    "XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0]))
}

# Training and Evaluation for Threshold Activation
for dataset_name, (X, y) in datasets.items():
    print(f"Training on {dataset_name} dataset with Threshold Activation Function")
    perceptron = PerceptronThreshold(input_size=X.shape[1])
    perceptron.train(X, y, epochs=100)
    accuracy, predictions = perceptron.evaluate(X, y)
    print(f"Accuracy on {dataset_name} with Threshold: {accuracy * 100}%")
    print(f"Predictions: {predictions}")
    print()

