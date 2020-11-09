"""
An example pipeline
"""
import os

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlinspect.utils import get_project_root

train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
train_data = pd.read_csv(train_file, na_values='?', index_col=0)
test_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_test.csv")
test_data = pd.read_csv(test_file, na_values='?', index_col=0)

train_labels = preprocessing.label_binarize(train_data['income-per-year'], classes=['>50K', '<=50K'])
test_labels = preprocessing.label_binarize(test_data['income-per-year'], classes=['>50K', '<=50K'])

nested_categorical_feature_transformation = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

nested_feature_transformation = ColumnTransformer(transformers=[
        ('categorical', nested_categorical_feature_transformation, ['education', 'workclass']),
        ('numeric', StandardScaler(), ['age', 'hours-per-week'])
    ])

nested_income_pipeline = Pipeline([
    ('features', nested_feature_transformation),
    ('classifier', DecisionTreeClassifier())])

nested_income_pipeline.fit(train_data, train_labels)

print(nested_income_pipeline.score(test_data, test_labels))
