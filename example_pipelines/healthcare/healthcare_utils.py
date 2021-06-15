"""
Some useful utils for the project
"""
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD


def create_model(input_dim=10):
    """Create a simple neural network"""
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=input_dim))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(2, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return clf
