# Support for different libraries and API functions

## Handling of unknown functions
* Extending mlinspect to support more and more API functions and libraries will be an ongoing effort. External contributions are very welcome! 
* However, mlinspect doesn't just crash when it encounters unknown functions.
* mlinspect just ignores functions it doesn't recognize. If a function it does recognize encounters the input from a relevant unknown function, it will create a `MISSING_OP` node for a single or multiple unknown function calls. The inspections also get to see this unknown input, from their perspective it's just a new data source.
* Example:
```python
import networkx
from inspect import cleandoc
from testfixtures import compare
from mlinspect import OperatorType, OperatorContext, FunctionInfo, PipelineInspector, CodeReference, DagNode, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo


test_code = cleandoc("""
        from inspect import cleandoc
        import pandas
        from mlinspect.testing._testing_helper_utils import black_box_df_op
        
        df = black_box_df_op()
        df = df.dropna()
        """)

extracted_dag = PipelineInspector.on_pipeline_from_string(test_code).execute().dag

expected_dag = networkx.DiGraph()
expected_missing_op = DagNode(-1,
                              BasicCodeLocation("<string-source>", 5),
                              OperatorContext(OperatorType.MISSING_OP, None),
                              DagNodeDetails('Warning! Operator <string-source>:5 (df.dropna()) encountered a '
                                             'DataFrame resulting from an operation without mlinspect support!',
                                             ['A']),
                              OptionalCodeInfo(CodeReference(5, 5, 5, 16), 'df.dropna()'))
expected_select = DagNode(0,
                          BasicCodeLocation("<string-source>", 5),
                          OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                          DagNodeDetails('dropna', ['A']),
                          OptionalCodeInfo(CodeReference(5, 5, 5, 16), 'df.dropna()'))
expected_dag.add_edge(expected_missing_op, expected_select)
compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))
```

## Pandas 
* The implementation can be found mainly [here](./_patch_pandas.py)
* The [tests](../../test/monkeypatching/test_patch_pandas.py) are probably more useful to look at
* Currently supported functions: 

| Function Call        | Operator        
| ------------- |:-------------:|
| `('pandas.io.parsers', 'read_csv')`      | Data Source | 
| `('pandas.core.frame', 'DataFrame')`      | Data Source      | 
| `('pandas.core.series', 'Series')`      | Data Source      | 
| `('pandas.core.frame', '__getitem__')`, arg type: strings | Projection|
| `('pandas.core.frame', '__getitem__')`, arg type: series | Selection |
| `('pandas.core.frame', 'dropna')` | Selection      |
| `('pandas.core.frame', 'replace')` | Projection (Mod)      |
| `('pandas.core.frame', '__setitem__')` | Projection (Mod)      |
| `('pandas.core.frame', 'merge')` | Join      |
| `('pandas.core.frame', 'groupby')` | Nothing (until a following agg call)     |
| `('pandas.core.groupbygeneric', 'agg')` | Groupby/Agg      |

## Sklearn 
* The implementation can be found mainly [here](./_patch_sklearn.py)
* The [tests](../../test/monkeypatching/test_patch_sklearn.py) are probably more useful to look at 
* Currently supported functions: 

| Function Call        | Operator        
| ------------- |:-------------:|
| `('sklearn.compose._column_transformer', 'ColumnTransformer')`, column selection      | Projection |
| `('sklearn.preprocessing._label', 'label_binarize')` | Projection (Mod)      |
| `('sklearn.compose._column_transformer', 'ColumnTransformer')`, concatenation      | Concatenation      |
| `('sklearn.model_selection._split', 'train_test_split')` | Split (Train/Test) 
| `('sklearn.preprocessing._encoders', 'OneHotEncoder')`, arg type: strings | Transformer |
| `('sklearn.preprocessing._data', 'StandardScaler')` | Transformer      |
| `('sklearn.impute._base’, 'SimpleImputer')` | Transformer      |
| `('sklearn.feature_extraction.text’, 'HashingVectorizer')` | Transformer      |
| `('sklearn.preprocessing._discretization', 'KBinsDiscretizer')` | Transformer      |
| `('sklearn.preprocessing_function_transformer','FunctionTransformer')` | Transformer      |
| `('sklearn.tree._classes', 'DecisionTreeClassifier')` | Estimator      |
| `('sklearn.linear_model._stochastic_gradient', 'SGDClassifier')` | Estimator      |
| `('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier')` | Estimator      |
| `('sklearn.linear_model._logistic', 'LogisticRegression')` | Estimator      |


## Numpy 
* The implementation can be found mainly [here](./_patch_numpy.py)
* The [tests](../../test/monkeypatching/test_patch_numpy.py) are probably more useful to look at 
* Currently supported functions: 

| Function Call        | Operator        
| ------------- |:-------------:|
| `('numpy.random', 'random')`      | Data Source | 

## Statsmodels
* The implementation can be found mainly [here](./_patch_statsmodels.py)
* The [tests](../../test/monkeypatching/test_patch_statsmodels.py) are probably more useful to look at 
* Currently supported functions: 

| Function Call        | Operator        
| ------------- |:-------------:|
| `('statsmodels.datasets', 'get_rdataset')`      | Data Source | 
| `('statsmodels.api', 'add_constant')`      | Projection (Mod) | 
| `('statsmodel.api', 'OLS')`, numpy syntax      | Estimator | 
