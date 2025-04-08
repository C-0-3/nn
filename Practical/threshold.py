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
            weights[0] += l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] += l_rate * error * row[i]
    return weights

# Logic gate datasets
AND_data = [[0,0,0],[1,0,0],[0,1,0],[1,1,1]]
OR_data  = [[0,0,0],[1,0,1],[0,1,1],[1,1,1]]
XOR_data = [[0,0,0],[1,0,1],[0,1,1],[1,1,0]]

# Training
l_rate = 0.2
n_epoch = 10
weights_AND = train_weights(AND_data, l_rate, n_epoch)
weights_OR  = train_weights(OR_data,  l_rate, n_epoch)
weights_XOR = train_weights(XOR_data, l_rate, n_epoch)

# Predictions
print("AND Gate Predictions:")
for row in AND_data:
    print(f"Input: {row[:2]}, Expected: {row[-1]}, Predicted: {int(predict(row, weights_AND))}")

print("\nOR Gate Predictions:")
for row in OR_data:
    print(f"Input: {row[:2]}, Expected: {row[-1]}, Predicted: {int(predict(row, weights_OR))}")

print("\nXOR Gate Predictions:")
for row in XOR_data:
    print(f"Input: {row[:2]}, Expected: {row[-1]}, Predicted: {int(predict(row, weights_XOR))}")
