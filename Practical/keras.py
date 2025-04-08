# pip install tensorflow
# pip install scikit-learn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical

data = load_iris()
x1 = data.data
y1 = to_categorical(data.target)

model = Sequential([
    Dense(12, input_shape = (4,), activation = 'relu'),
    Dense(8, activation = 'relu'),
    Dense(3, activation = 'softmax')
])

model.compile(loss = 'crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x1, y1, epochs = 100, batch_size = 10)

_, accuracy = model.evaluate(x1, y1)
print("Accuracy: %.2f" % (accuracy * 100))
