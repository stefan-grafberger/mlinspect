"""
An example pipeline
"""
import os
import pandas as pd

from sklearn import compose, preprocessing, tree, pipeline
from mlinspect.utils import get_project_root

print('pipeline start')
train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
raw_data = pd.read_csv(train_file, na_values='?', index_col=0)

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
