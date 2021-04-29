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

train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_demo", "adult_demo.csv")
train_data = pd.read_csv(train_file, na_values='?', index_col=0)

# print(train_data)
#occupation_notna = train_data['occupation'].notna()
#native_country_notna = train_data['native-country'].notna()
#income_notna = train_data['income-per-year'].notna()
#train_data = train_data[occupation_notna & native_country_notna & income_notna]
#train_data = train_data[train_data['occupation'].notna()]
train_data = train_data[train_data['native-country'].notna()]
#train_data = train_data[train_data['income-per-year'].notna()]
# print(train_data)

train_labels = preprocessing.label_binarize(train_data['income-per-year'], classes=['>50K', '<=50K'])

train_series = pd.Series(np.squeeze(train_labels))
# print(train_series)

test = train_data.corrwith(train_series)
# print(test)

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

# print(nested_income_pipeline.score(test_data, test_labels))
