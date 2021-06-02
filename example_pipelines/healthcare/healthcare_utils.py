"""
Some useful utils for the project
"""
import numpy
from sklearn.exceptions import NotFittedError
from gensim.sklearn_api import W2VTransformer
from tensorflow.keras.layers import Dense  # pylint: disable=no-name-in-module
from tensorflow.keras.models import Sequential  # pylint: disable=no-name-in-module
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD  # pylint: disable=no-name-in-module
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier  # pylint: disable=no-name-in-module


class MyW2VTransformer(W2VTransformer):
    """Some custom w2v transformer."""

    def __init__(self, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5,
                 null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000):
        super().__init__(size=size, alpha=alpha, window=window,
                         min_count=min_count, max_vocab_size=max_vocab_size, sample=sample,
                         seed=seed, workers=workers, min_alpha=min_alpha, sg=sg, hs=hs,
                         negative=negative, cbow_mean=cbow_mean, hashfxn=hashfxn, iter=iter,
                         null_word=null_word, trim_rule=trim_rule, sorted_vocab=sorted_vocab,
                         batch_words=batch_words)

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


class MyKerasClassifier(KerasClassifier):
    """A Keras Wrapper that sets input_dim on fit"""

    def __init__(self, build_fn=None, **sk_params):
        super().__init__(build_fn, **sk_params)

    def fit(self, x, y, **kwargs):
        """Create and fit a simple neural network"""
        self.sk_params['input_dim'] = x.shape[1]
        super().fit(x, y, **kwargs)


def create_model(input_dim=10):
    """Create a simple neural network"""
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=input_dim))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(2, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return clf
