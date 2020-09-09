"""
An example pipeline
"""
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from demo.healthcare.demo_utils import MyW2VTransformer, create_model
from mlinspect.utils import get_project_root

COUNTIES_OF_INTEREST = ['county2', 'county3']

# load input data sources (data generated with https://www.mockaroo.com as a single file and then split into two)
patients = pd.read_csv(os.path.join(str(get_project_root()), "demo", "healthcare", "healthcare_patients.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "demo", "healthcare", "healthcare_histories.csv"),
                        na_values='?')

# combine input data into a single table
data = patients.merge(histories, on=['ssn'])

# compute mean complications per age group, append as column
complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))

data = data.merge(complications, on=['age_group'])

# target variable: people with a high number of complications
data['label'] = data['complications'] > 1.2 * data['mean_complications']

# project data to a subset of attributes
data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]

# filter data
data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

# define the feature encoding of the data
impute_and_one_hot_encode = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])

featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    ('word2vec', MyW2VTransformer(min_count=2), ['last_name']),
    ('numeric', StandardScaler(), ['num_children', 'income'])
])

# define the training pipeline for the model
neural_net = KerasClassifier(build_fn=create_model, epochs=10, batch_size=1, verbose=0, input_dim=109)
pipeline = Pipeline([
    ('features', featurisation),
    ('learner', neural_net)])

# train-test split
train_data, test_data = train_test_split(data, random_state=0)
# model training
model = pipeline.fit(train_data, train_data['label'])
# model evaluation
print(model.score(test_data, test_data['label']))
