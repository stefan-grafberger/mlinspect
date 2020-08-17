"""
An example pipeline
"""
import os

import pandas as pd
from gensim.sklearn_api import W2VTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from mlinspect.utils import get_project_root

COUNTIES_OF_INTEREST = ['county1', 'county2', 'county3']


# create a function that returns a model, taking as parameters things you
# want to verify using cross-valdiation and model selection
def create_model(optimizer='adagrad',
                 kernel_initializer='glorot_uniform',
                 dropout=0.2):
    nn_model = Sequential()
    nn_model.add(Dense(64, activation='relu', kernel_initializer=kernel_initializer))
    nn_model.add(Dropout(dropout))
    nn_model.add(Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer))

    nn_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return nn_model


# load input data sources
patients = pd.read_csv(os.path.join(str(get_project_root()), "test", "data", "adult_train.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "test", "data", "adult_train.csv"), na_values='?')

# combine input data into a single table
data = patients.merge(histories, on=['ssn'])

# compute mean complications per age group, append as column
complications = data.groupby('age_group')['complications'].mean().rename({'complications': 'mean_complications'})
data = data.merge(complications, on=['age_group'])

# target variable: people with a high number of complications
data['label'] = data['complications'] > 2 * data['mean_complications']

# project data to a subset of attributes
data = data[['smoker', 'family_name', 'county', 'num_children', 'race', 'income', 'label']]

# filter data
data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

# define the feature encoding of the data
impute_and_one_hot_encode = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder())
    ])

featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    ('word2vec', W2VTransformer(size=10), ['family_name']),
    ('numeric', StandardScaler(), ['num_children', 'income'])
])

# define the training pipeline for the model
neural_net = KerasClassifier(build_fn=create_model())
pipeline = Pipeline([
    ('features', featurisation),
    ('learner', neural_net)])

# train-test split
train_data, test_data = train_test_split(data)
# model training
model = pipeline.fit(train_data, train_data['label'])
# model evaluation
print(model.score(test_data, test_data['label']))
