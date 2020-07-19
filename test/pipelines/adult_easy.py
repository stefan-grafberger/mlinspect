import pandas as pd

from sklearn import compose, preprocessing, tree, pipeline

print('Hello World')
raw_data = pd.read_csv('/Users/stefangrafberger/Documents/uni/master-thesis/fairDAGs-master/data/adult_train.csv', na_values='?', index_col=0)

data = raw_data.dropna()

labels = preprocessing.label_binarize(data['income-per-year'], ['>50K', '<=50K'])

feature_transformation = compose.ColumnTransformer(transformers=[
    ('categorical', preprocessing.OneHotEncoder(handle_unknown='ignore'), ['education', 'workclass']),
    ('numeric', preprocessing.StandardScaler(), ['age', 'hours-per-week'])
])


income_pipeline = pipeline.Pipeline([
    ('features', feature_transformation),
    ('classifier', tree.DecisionTreeClassifier())])

income_pipeline.fit(data, labels)
print(income_pipeline.predict(data))