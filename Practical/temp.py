def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

def train_weights(dataset, l_rate, n_epoch):
    weights = [0.0 for _ in range(len(dataset[0]))]
    for epoch in range(n_epoch):
        for row in dataset:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights += l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] += l_rate * error * row[i]