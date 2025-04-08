#tanh
import numpy as np

def tanh(x):
    return np.tanh(x)

def predict_tanh(row, weight):
    activation = weight[0]
    for i in range(len(row) - 1):
        activation += weight[i + 1] * row[i]
    return tanh(activation)

def train_tanh_weight(train, l_rate, n_epoch):
    weight = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict_tanh(row, weight)
            error = row[-1] - prediction
            sum_error += error**2
            weight[0] += l_rate * error
            for i in range(len(row) - 1):
                weight[i + 1] += l_rate * error * row[i]
        print(">epoch = %d, l_rate = %.3f, error = %.3f" % (epoch, l_rate, sum_error))
    return weight

# Logic gate datasets
dataset_AND = [[0, 0, 0],
               [1, 0, 0],
               [0, 1, 0],
               [1, 1, 1]]

dataset_OR = [[0, 0, 0],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 1]]

dataset_XOR = [[0, 0, 0],
               [1, 0, 1],
               [0, 1, 1],
               [1, 1, 0]]

# Hyperparameters
l_rate = 0.2
n_epoch = 10

# Train weights
weight_AND = train_tanh_weight(dataset_AND, l_rate, n_epoch)
weight_OR = train_tanh_weight(dataset_OR, l_rate, n_epoch)
weight_XOR = train_tanh_weight(dataset_XOR, l_rate, n_epoch)

# Print predictions
print("\nAND GATE TANH")
for row in dataset_AND:
    prediction = predict_tanh(row, weight_AND)
    print("EXPECTED = %d , PREDICTED = %.3f" % (row[-1], prediction))

print("\nOR GATE TANH")
for row in dataset_OR:
    prediction = predict_tanh(row, weight_OR)
    print("EXPECTED = %d , PREDICTED = %.3f" % (row[-1], prediction))

print("\nXOR GATE TANH")
for row in dataset_XOR:
    prediction = predict_tanh(row, weight_XOR)
    print("EXPECTED = %d , PREDICTED = %.3f" % (row[-1], prediction))