"""
An example pipeline
"""
import os
import pandas as pd

from sklearn import compose, preprocessing, tree, pipeline
from mlinspect.utils import get_project_root

print('pipeline start')

train_file_a =os.path.join(str(get_project_root()), "experiments", "user_interviews", "adult_simple_train_a.csv")
raw_data_a = pd.read_csv(train_file_a, na_values='?', index_col=0)

train_file_b = os.path.join(str(get_project_root()), "experiments", "user_interviews", "adult_simple_train_b.csv")
raw_data_b = pd.read_csv(train_file_b, na_values='?', index_col=0)

raw_data = raw_data_a.merge(raw_data_b, on="id")

data = raw_data.dropna()

labels = preprocessing.label_binarize(data['income-per-year'], classes=['>50K', '<=50K'])

feature_transformation = compose.ColumnTransformer(transformers=[
    ('categorical', preprocessing.OneHotEncoder(handle_unknown='ignore'), ['education', 'workclass']),
    ('numeric', preprocessing.StandardScaler(), ['age', 'hours-per-week'])
])


income_pipeline = pipeline.Pipeline([
    ('features', feature_transformation),
    ('classifier', tree.DecisionTreeClassifier())])

income_pipeline.fit(data, labels)


print('pipeline finished')
