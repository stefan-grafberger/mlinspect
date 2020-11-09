"""
COMPAS pipeline
"""
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, label_binarize

from mlinspect.utils import get_project_root

train_file = os.path.join(str(get_project_root()), "experiments", "user_interviews", "compas_train_modified.csv")
train = pd.read_csv(train_file, na_values='?', index_col=0)
test_file = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas_test.csv")
test = pd.read_csv(test_file, na_values='?', index_col=0)

train = train[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
test = test[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

# If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested,
# we assume that because of data quality reasons, that we do not have the right offense.
train = train[(train['days_b_screening_arrest'] <= 30) & (train['days_b_screening_arrest'] >= -30)]
# We coded the recidivist flag – is_recid – to be -1 if we could not find a compas case at all.
train = train[train['is_recid'] != -1]
# In a similar vein, ordinary traffic offenses – those with a c_charge_degree of ‘O’ – will not result in Jail
# time are removed (only two of them).
train = train[train['c_charge_degree'] != "O"]
# We filtered the underlying data from Broward county to include only those rows representing people who had either
# recidivated in two years, or had at least two years outside of a correctional facility.
train = train[train['score_text'] != 'N/A']

train = train.replace('Medium', "Low")
test = test.replace('Medium', "Low")

train_labels = label_binarize(train['score_text'], classes=['High', 'Low'])
test_labels = label_binarize(test['score_text'], classes=['High', 'Low'])

impute_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                              ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                           ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

compas_featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute_and_bin, ['age'])
])
compas_pipeline = Pipeline([
    ('features', compas_featurizer),
    ('classifier', LogisticRegression())
])

compas_pipeline.fit(train, train_labels.ravel())
print(compas_pipeline.score(test, test_labels.ravel()))
