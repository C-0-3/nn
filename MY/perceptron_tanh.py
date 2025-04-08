import numpy as np
import matplotlib.pyplot as plt

# Tanh activation function
def tanh_activation(x):
    return np.tanh(x)

# Derivative of Tanh for backpropagation
def tanh_derivative(x):
    return 1.0 - np.square(np.tanh(x))

# Perceptron class for Tanh activation function
class PerceptronTanh:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size + 1)  # weights + bias
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return tanh_activation(summation)

    def train(self, inputs, targets, epochs=100):
        for epoch in range(epochs):
            for input_data, target in zip(inputs, targets):
                prediction = self.predict(input_data)
                error = target - prediction
                self.weights[1:] += self.learning_rate * error * tanh_derivative(prediction) * input_data
                self.weights[0] += self.learning_rate * error * tanh_derivative(prediction)

    def evaluate(self, inputs, targets):
        predictions = [self.predict(input_data) for input_data in inputs]
        accuracy = np.mean(np.round(np.array(predictions)) == np.array(targets))
        return accuracy, predictions


# Training and Evaluation for Tanh Activation
for dataset_name, (X, y) in datasets.items():
    print(f"Training on {dataset_name} dataset with Tanh Activation Function")
    perceptron = PerceptronTanh(input_size=X.shape[1])
    perceptron.train(X, y, epochs=100)
    accuracy, predictions = perceptron.evaluate(X, y)
    print(f"Accuracy on {dataset_name} with Tanh: {accuracy * 100}%")
    print(f"Predictions: {np.round(predictions)}")
    print()

