"""
An example pipeline
"""
import os
import pandas as pd

from sklearn import compose, preprocessing, tree, pipeline
from mlinspect.utils import get_project_root

COUNTIES_OF_INTEREST = ['county1', 'county2', 'county3']

# load input data sources
patients = os.path.join(str(get_project_root()), "test", "data", "adult_train.csv")
histories = os.path.join(str(get_project_root()), "test", "data", "adult_train.csv")

# combine input data into a single table
data = pd.merge([patients, histories], on=['ssn'])

# compute mean complications per age group, append as column
complications = data.groupby('age_group')['complications'].mean().rename({'complications': 'mean_complications'})
data = pd.merge([data, complications], on=['age_group'])

# target variable: people with a high number of complications
data['label'] = data['complications'] > 2 * data['mean_complications']

# project data to a subset of attributes
data = data[['smoker', 'family_name', 'county', 'num_children', 'race', 'income', 'label']]

# filter data
data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

# FIXME

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
