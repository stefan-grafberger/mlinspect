"""
Some useful utils for the project
"""
import numpy
from keras import Input
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from example_pipelines.healthcare._gensim_wrapper import W2VTransformer


class MyW2VTransformer(W2VTransformer):
    """Some custom w2v transformer."""
    # pylint: disable-all

    def partial_fit(self, X):
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


def create_model(meta, hidden_layer_sizes):
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]
    model = Sequential()
    model.add(Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(Dense(hidden_layer_size, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model
