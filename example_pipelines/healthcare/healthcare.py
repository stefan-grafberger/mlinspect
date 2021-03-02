"""
An example pipeline
"""
import os

import pandas as pd
from datawig import SimpleImputer as DatawigSimpleImputer

from mlinspect.utils import get_project_root

# load input data sources (data generated with https://www.mockaroo.com as a single file and then split into two)
patients = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                    "healthcare_patients.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "example_pipelines", "healthcare",
                                     "healthcare_histories.csv"), na_values='?')

# combine input data into a single table
data = patients.merge(histories, on=['ssn'])

# project data to a subset of attributes
data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income']]

# we have rows with missing values now
print(data[data.isnull().any(axis=1)])

# imputing
smoker_imputer = DatawigSimpleImputer(
    input_columns=['county', 'num_children', 'income'],
    output_column='smoker',
    output_path='imputer_model'
)
smoker_imputer.fit(train_df=data, num_epochs=5)
data = smoker_imputer.predict(data)
data["smoker"] = data["smoker_imputed"]

race_imputer = DatawigSimpleImputer(
    input_columns=['county', 'num_children', 'income'],
    output_column='race',
    output_path='imputer_model'
)
race_imputer.fit(train_df=data, num_epochs=5)
data = race_imputer.predict(data)
data["race"] = data["race_imputed"]

county_imputer = DatawigSimpleImputer(
    input_columns=['race', 'num_children', 'income'],
    output_column='county',
    output_path='imputer_model'
)
county_imputer.fit(train_df=data, num_epochs=5)
data = county_imputer.predict(data)
data["county"] = data["county_imputed"]

data = data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income']]

# no rows should now have missing values
#print(data[data.isnull().any(axis=1)])
