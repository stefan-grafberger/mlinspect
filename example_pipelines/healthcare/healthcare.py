import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer, MyKerasClassifier, create_model
from mlinspect.utils import get_project_root

patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                    "healthcare_patients.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                     "healthcare_histories.csv"), na_values='?')

counties_to_change = patients[patients['county'].isin(['county1', 'county3'])].sample(frac=0.2).index
patients.loc[counties_to_change, 'county'] = 'county4'
patients.to_csv(path_or_buf=os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "histories.csv"))
histories.to_csv(path_or_buf=os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "patients.csv"))


data = patients.merge(histories, on=['ssn'])
complications = data.groupby('age_group').agg(mean_complications=('complications', 'mean'))
data = data.merge(complications, on=['age_group'])
data['label'] = data['complications'] > 1.2 * data['mean_complications']
data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]

data = data[data['county'].isin(['county1', 'county3'])]
#data = data[data['county'].isin(['county1', 'county3', 'county2'])]  # Replacement fixes NoBiasIntroducedFor issue in line above

impute_and_one_hot_encode = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])

featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    ('word2vec', MyW2VTransformer(min_count=2), ['last_name']),
    ('numeric', StandardScaler(), ['num_children', 'income'])
])

neural_net = MyKerasClassifier(build_fn=create_model, epochs=10, batch_size=1, verbose=0)
pipeline = Pipeline([
    ('features', featurisation),
    ('learner', neural_net)])

train_data, test_data = train_test_split(data)
model = pipeline.fit(train_data, train_data['label'])
print(model.score(test_data, test_data['label']))
