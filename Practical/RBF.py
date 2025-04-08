import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

x, y = load_wine(return_X_y = True)
x = MinMaxScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

def rbf_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.dot(x1 - x2, (x1 - x2).T))

def rbf_classifier(x_train, y_train, x_test, gamma):
    return np.array([
        y_train[np.argmax([rbf_kernel(x, train, gamma) for train in x_train])]
        for x in x_test
    ])

predictions = rbf_classifier(x_train, y_train, x_test, gamma = 0.5)
print("Accuracy: %.2f" % (accuracy_score(y_test, predictions) * 100))
