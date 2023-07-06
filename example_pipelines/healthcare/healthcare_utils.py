"""
Some useful utils for the project
"""
import numpy
from sklearn.exceptions import NotFittedError
from gensim.sklearn_api import W2VTransformer
from tensorflow.keras.layers import Dense  # pylint: disable=no-name-in-module
from tensorflow.keras.models import Sequential  # pylint: disable=no-name-in-module
from keras.optimizers.optimizer_v2.gradient_descent import SGD  # pylint: disable=no-name-in-module


class MyW2VTransformer(W2VTransformer):
    """Some custom w2v transformer."""

    def partial_fit(self, X):
        # pylint: disable=useless-super-delegation
        super().partial_fit([X])

    def fit(self, X, y=None):
        X = X.iloc[:, 0].tolist()
        return super().fit([X], y)

    def transform(self, words):
        words = words.iloc[:, 0].tolist()
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        vectors = []
        for word in words:
            if word in self.gensim_model.wv:
                vectors.append(self.gensim_model.wv[word])
            else:
                vectors.append(numpy.zeros(self.size))
        return numpy.reshape(numpy.array(vectors), (len(words), self.size))


def create_model():
    """Create a simple neural network"""
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=109))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    clf.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return clf

def create_model_with_input(input_dim=10):
    """Create a simple neural network"""
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=input_dim))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(2, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return clf
