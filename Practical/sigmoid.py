import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_sigmoid(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return sigmoid(activation)

def train_sigmoid_weights(dataset, l_rate, n_epoch):
    weights = [0.0 for _ in range(len(dataset[0]))]
    for epoch in range(n_epoch):
        for row in dataset:
            prediction = predict_sigmoid(row, weights)
            error = row[-1] - prediction
            weights[0] += l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] += l_rate * error * row[i]
        print(f">epoch={epoch}, l_rate={l_rate:.3f}, error={error:.3f}")
    return weights

# Logic gate datasets
dataset_AND = [[0,0,0],[1,0,0],[0,1,0],[1,1,1]]
dataset_OR  = [[0,0,0],[1,0,1],[0,1,1],[1,1,1]]
dataset_XOR = [[0,0,0],[1,0,1],[0,1,1],[1,1,0]]

# Training
l_rate = 0.2
epochs = 10
weights_AND = train_sigmoid_weights(dataset_AND, l_rate, epochs)
weights_OR  = train_sigmoid_weights(dataset_OR,  l_rate, epochs)
weights_XOR = train_sigmoid_weights(dataset_XOR, l_rate, epochs)

# Predictions
print("\nAND Gate with Sigmoid:")
for row in dataset_AND:
    pred = predict_sigmoid(row, weights_AND)
    print(f"Input: {row[:2]}, Expected: {row[-1]}, Predicted: {pred:.3f}")

print("\nOR Gate with Sigmoid:")
for row in dataset_OR:
    pred = predict_sigmoid(row, weights_OR)
    print(f"Input: {row[:2]}, Expected: {row[-1]}, Predicted: {pred:.3f}")

print("\nXOR Gate with Sigmoid:")
for row in dataset_XOR:
    pred = predict_sigmoid(row, weights_XOR)
    print(f"Input: {row[:2]}, Expected: {row[-1]}, Predicted: {pred:.3f}")
