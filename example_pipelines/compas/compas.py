"""
An example pipeline
"""
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, label_binarize

from mlinspect.utils import get_project_root

train_file = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas_train.csv")
train_data = pd.read_csv(train_file, na_values='?', index_col=0)
test_file = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas_test.csv")
test_data = pd.read_csv(test_file, na_values='?', index_col=0)

train_data = train_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
test_data = test_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

train_data = train_data[(train_data['days_b_screening_arrest'] <= 30) & (train_data['days_b_screening_arrest'] >= -30)]
train_data = train_data[train_data['is_recid'] != -1]
train_data = train_data[train_data['c_charge_degree'] != "O"]
train_data = train_data[train_data['score_text'] != 'N/A']

train_data = train_data.replace('Medium', "Low")
test_data = test_data.replace('Medium', "Low")

train_labels = label_binarize(train_data['score_text'], classes=['High', 'Low'])
test_labels = label_binarize(test_data['score_text'], classes=['High', 'Low'])

impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute2_and_bin, ['age'])
])

pipeline = Pipeline([
    ('features', featurizer),
    ('classifier', LogisticRegression())
])

pipeline.fit(train_data, train_labels.ravel())
print(pipeline.score(test_data, test_labels.ravel()))
