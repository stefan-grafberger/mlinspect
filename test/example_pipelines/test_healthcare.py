"""
Tests whether the healthcare demo works
"""
import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from example_pipelines.healthcare import custom_monkeypatching
from example_pipelines.healthcare.healthcare_utils import MyKerasClassifier, create_model, MyW2VTransformer
from example_pipelines import HEALTHCARE_PY, HEALTHCARE_PNG
from mlinspect.testing._testing_helper_utils import run_and_assert_all_op_outputs_inspected


def test_my_word_to_vec_transformer():
    """
    Tests whether MyW2VTransformer works
    """
    pandas_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
    word_to_vec = MyW2VTransformer(min_count=2, size=2, workers=1)
    encoded_data = word_to_vec.fit_transform(pandas_df)
    assert encoded_data.shape == (4, 2)


def test_my_keras_classifier():
    """
    Tests whether MyKerasClassifier works
    """
    pandas_df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

    train = StandardScaler().fit_transform(pandas_df[['A', 'B']])
    target = OneHotEncoder(sparse=False).fit_transform(pandas_df[['target']])

    clf = MyKerasClassifier(build_fn=create_model, epochs=2, batch_size=1, verbose=0)
    clf.fit(train, target)

    test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
    assert test_predict.shape == (2,)


def test_py_pipeline_runs():
    """
    Tests whether the pipeline works without instrumentation
    """
    with open(HEALTHCARE_PY) as file:
        healthcare_code = file.read()
        parsed_ast = ast.parse(healthcare_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"))


def test_instrumented_py_pipeline_runs():
    """
    Tests whether the pipeline works with instrumentation
    """
    dag = run_and_assert_all_op_outputs_inspected(HEALTHCARE_PY, ["age_group", "race"], HEALTHCARE_PNG,
                                                  [custom_monkeypatching])
    assert len(dag) == 37
