"""
An example pipeline
"""
import os

import pandas as pd
from gensim.models import Word2Vec
from gensim.sklearn_api import W2VTransformer
from gensim.test.utils import common_texts
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from zeugma.embeddings import EmbeddingTransformer

from mlinspect.utils import get_project_root

COUNTIES_OF_INTEREST = ['Iowa', 'Florida', 'Ohio', 'California', 'Nevada', 'Texas', 'New York', 'Missouri', 'Virginia']


# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection
def create_model():
    """Create a simple neural network"""
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim=2))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
    return clf


# load input data sources (data generated with https://www.mockaroo.com as a single file and then split into two)
patients = pd.read_csv(os.path.join(str(get_project_root()), "test", "data", "healthcare_patients.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "test", "data", "healthcare_histories.csv"),
                        na_values='?')

# combine input data into a single table
data = patients.merge(histories, on=['ssn'])

# compute mean complications per age group, append as column
complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean')).reset_index()

data = data.merge(complications, on=['age_group'])

# target variable: people with a high number of complications
data['label'] = data['complications'] > 1.2 * data['mean_complications']

# project data to a subset of attributes
data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]

# filter data
data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

# define the feature encoding of the data
impute_and_one_hot_encode = Pipeline(steps=[
        #('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder())
    ])

#w2v_transformer = W2VTransformer(min_count=0, seed=1)

#last_name_list = data['last_name'].tolist()
#last_name_list.append('last_name')
#last_name_list = [[name] for name in last_name_list]
#w2v_transformer = w2v_transformer.fit(last_name_list)
#w2v = Word2Vec(last_name_list)
#w2v.train(last_name_list)

# glove = EmbeddingTransformer('glove-wiki-gigaword-50')

featurisation = ColumnTransformer(transformers=[
    #("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    #('word2vec', OneHotEncoder(), ['last_name']),
    ('numeric', StandardScaler(), ['num_children', 'income'])
])

# define the training pipeline for the model
neural_net = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=0)
pipeline = Pipeline([
    ('features', featurisation),
    ('learner', DecisionTreeClassifier())])

# train-test split
train_data, test_data = train_test_split(data)
# model training
model = pipeline.fit(train_data, train_data['label'])
# model evaluation
print(model.score(test_data, test_data['label']))
