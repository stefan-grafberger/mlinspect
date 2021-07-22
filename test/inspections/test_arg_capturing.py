"""
Tests whether ArgumentCapturing works
"""
from inspect import cleandoc

import numpy
from testfixtures import compare

from mlinspect import OperatorType, DagNode, BasicCodeLocation, OperatorContext, FunctionInfo, DagNodeDetails, \
    OptionalCodeInfo, CodeReference
from mlinspect.inspections import ArgumentCapturing
from mlinspect.instrumentation import _pipeline_executor


def test_arg_capturing_sklearn_decision_tree():
    """
    Tests whether ArgumentCapturing works for the sklearn DecisionTreeClassifier
    """
    test_code = cleandoc("""
                    import pandas as pd
                    from sklearn.preprocessing import label_binarize, StandardScaler
                    from sklearn.tree import DecisionTreeClassifier
                    import numpy as np

                    df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                    train = StandardScaler().fit_transform(df[['A', 'B']])
                    target = label_binarize(df['target'], classes=['no', 'yes'])

                    clf = DecisionTreeClassifier()
                    clf = clf.fit(train, target)

                    test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'target': ['no', 'yes']})
                    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
                    test_score = clf.score(test_df[['A', 'B']], test_labels)
                    assert test_score == 1.0
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    classifier_node = list(inspector_result.dag.nodes)[7]
    score_node = list(inspector_result.dag.nodes)[14]

    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                  DagNodeDetails('Decision Tree', []),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 30),
                                                   'DecisionTreeClassifier()'))
    expected_score = DagNode(14,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier', 'score')),
                             DagNodeDetails('Decision Tree', []),
                             OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                              "clf.score(test_df[['A', 'B']], test_labels)"))

    compare(classifier_node, expected_classifier)
    compare(score_node, expected_score)

    expected_args = {'criterion': 'gini', 'splitter': 'best', 'max_depth': None, 'min_samples_split': 2,
                     'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': None, 'random_state': None,
                     'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None,
                     'class_weight': None, 'presort': 'deprecated', 'ccp_alpha': 0.0}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[classifier_node]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[score_node]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)


def test_arg_capturing_sklearn_sgd_classifier():
    """
    Tests whether ArgumentCapturing works for the sklearn SGDClassifier
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import SGDClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = SGDClassifier(loss='log', random_state=42)
                clf = clf.fit(train, target)

                test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'target': ['no', 'yes']})
                test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
                test_score = clf.score(test_df[['A', 'B']], test_labels)
                assert test_score == 1.0
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    classifier_node = list(inspector_result.dag.nodes)[7]
    score_node = list(inspector_result.dag.nodes)[14]

    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                               'SGDClassifier')),
                                  DagNodeDetails('SGD Classifier', []),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 48),
                                                   "SGDClassifier(loss='log', random_state=42)"))
    expected_score = DagNode(14,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.linear_model._stochastic_gradient.SGDClassifier',
                                                          'score')),
                             DagNodeDetails('SGD Classifier', []),
                             OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                              "clf.score(test_df[['A', 'B']], test_labels)"))

    compare(classifier_node, expected_classifier)
    compare(score_node, expected_score)

    expected_args = {'loss': 'log', 'penalty': 'l2', 'alpha': 0.0001, 'l1_ratio': 0.15, 'fit_intercept': True,
                     'max_iter': 1000, 'tol': 0.001, 'shuffle': True, 'verbose': 0, 'epsilon': 0.1, 'n_jobs': None,
                     'random_state': 42, 'learning_rate': 'optimal', 'eta0': 0.0, 'power_t': 0.5,
                     'early_stopping': False, 'validation_fraction': 0.1, 'n_iter_no_change': 5, 'class_weight': None,
                     'warm_start': False, 'average': False}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[classifier_node]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[score_node]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)


def test_arg_capturing_sklearn_keras_classifier():
    """
    Tests whether ArgumentCapturing works for the sklearn KerasClassifier
    """
    test_code = cleandoc("""
                    import pandas as pd
                    from sklearn.preprocessing import StandardScaler, OneHotEncoder
                    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
                    from tensorflow.keras.layers import Dense
                    from tensorflow.keras.models import Sequential
                    from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
                    import tensorflow as tf
                    import numpy as np
    
                    df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    
                    train = StandardScaler().fit_transform(df[['A', 'B']])
                    target = OneHotEncoder(sparse=False).fit_transform(df[['target']])
                    
                    def create_model(input_dim):
                        clf = Sequential()
                        clf.add(Dense(2, activation='relu', input_dim=input_dim))
                        clf.add(Dense(2, activation='relu'))
                        clf.add(Dense(2, activation='softmax'))
                        clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                        return clf
    
                    np.random.seed(42)
                    tf.random.set_seed(42)
                    clf = KerasClassifier(build_fn=create_model, epochs=15, batch_size=1, verbose=0, input_dim=2)
                    clf = clf.fit(train, target)
    
                    test_df = pd.DataFrame({'A': [0., 0.8], 'B':  [0., 0.8], 'target': ['no', 'yes']})
                    test_labels = OneHotEncoder(sparse=False).fit_transform(test_df[['target']])
                    test_score = clf.score(test_df[['A', 'B']], test_labels)
                    assert test_score == 1.0
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    classifier_node = list(inspector_result.dag.nodes)[7]
    score_node = list(inspector_result.dag.nodes)[14]

    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 25),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails('Neural Network', []),
                                  OptionalCodeInfo(CodeReference(25, 6, 25, 93),
                                                   'KerasClassifier(build_fn=create_model, epochs=15, batch_size=1, '
                                                   'verbose=0, input_dim=2)'))
    expected_score = DagNode(14,
                             BasicCodeLocation("<string-source>", 30),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.'
                                                          'KerasClassifier', 'score')),
                             DagNodeDetails('Neural Network', []),
                             OptionalCodeInfo(CodeReference(30, 13, 30, 56),
                                              "clf.score(test_df[['A', 'B']], test_labels)"))

    compare(classifier_node, expected_classifier)
    compare(score_node, expected_score)

    expected_args = {'epochs': 15, 'batch_size': 1, 'verbose': 0, 'input_dim': 2}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[classifier_node]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[score_node]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)


def test_arg_capturing_standard_scaler():
    """
    Tests whether ArgumentCapturing works for the sklearn StandardScaler
    """
    test_code = cleandoc("""
                    import pandas as pd
                    from sklearn.preprocessing import StandardScaler
                    import numpy as np
    
                    df = pd.DataFrame({'A': [1, 2, 10, 5]})
                    standard_scaler = StandardScaler()
                    encoded_data = standard_scaler.fit_transform(df)
                    test_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                    encoded_data = standard_scaler.transform(test_df)
                    expected = np.array([[-1.], [-0.71428571], [1.57142857], [0.14285714]])
                    assert np.allclose(encoded_data, expected)
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]

    expected_fit_transform = DagNode(1,
                                     BasicCodeLocation("<string-source>", 6),
                                     OperatorContext(OperatorType.TRANSFORMER,
                                                     FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                     DagNodeDetails('Standard Scaler: fit_transform', ['array']),
                                     OptionalCodeInfo(CodeReference(6, 18, 6, 34), 'StandardScaler()'))
    expected_transform = DagNode(3,
                                 BasicCodeLocation("<string-source>", 6),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                 DagNodeDetails('Standard Scaler: transform', ['array']),
                                 OptionalCodeInfo(CodeReference(6, 18, 6, 34), 'StandardScaler()'))

    compare(fit_transform_node, expected_fit_transform)
    compare(transform_node, expected_transform)

    expected_args = {'copy': True, 'with_mean': True, 'with_std': True}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_fit_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)


def test_arg_capturing_hashing_vectorizer():
    """
    Tests whether ArgumentCapturing works for the sklearn HasingVectorizer
    """
    test_code = cleandoc("""
                    import pandas as pd
                    from sklearn.feature_extraction.text import HashingVectorizer
                    from scipy.sparse import csr_matrix
                    import numpy as np
    
                    df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                    vectorizer = HashingVectorizer(ngram_range=(1, 3), n_features=2**2)
                    encoded_data = vectorizer.fit_transform(df['A'])
                    expected = csr_matrix([[-0., 0., 0., -1.], [0., -1., -0., 0.], [0., 0., 0., -1.], [0., 0., 0., -1.]])
                    assert np.allclose(encoded_data.A, expected.A)
                    test_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                    encoded_data = vectorizer.transform(test_df['A'])
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    fit_transform_node = list(inspector_result.dag.nodes)[2]
    transform_node = list(inspector_result.dag.nodes)[5]

    expected_fit_transform = DagNode(2,
                                     BasicCodeLocation("<string-source>", 7),
                                     OperatorContext(OperatorType.TRANSFORMER,
                                                     FunctionInfo('sklearn.feature_extraction.text',
                                                                  'HashingVectorizer')),
                                     DagNodeDetails('Hashing Vectorizer: fit_transform', ['array']),
                                     OptionalCodeInfo(CodeReference(7, 13, 7, 67),
                                                      'HashingVectorizer(ngram_range=(1, 3), n_features=2**2)'))
    expected_transform = DagNode(5,
                                 BasicCodeLocation("<string-source>", 7),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('sklearn.feature_extraction.text',
                                                              'HashingVectorizer')),
                                 DagNodeDetails('Hashing Vectorizer: transform', ['array']),
                                 OptionalCodeInfo(CodeReference(7, 13, 7, 67),
                                                  'HashingVectorizer(ngram_range=(1, 3), n_features=2**2)'))

    compare(fit_transform_node, expected_fit_transform)
    compare(transform_node, expected_transform)

    expected_args = {'input': 'content', 'encoding': 'utf-8', 'decode_error': 'strict', 'strip_accents': None,
                     'lowercase': True, 'preprocessor': None, 'tokenizer': None, 'stop_words': None,
                     'token_pattern': '(?u)\\b\\w\\w+\\b', 'ngram_range': (1, 3), 'analyzer': 'word', 'n_features': 4,
                     'binary': False, 'norm': 'l2', 'alternate_sign': True, 'dtype': numpy.float64}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_fit_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)


def test_arg_capturing_kbins_discretizer():
    """
    Tests whether ArgumentCapturing works for the sklearn KBinsDiscretizer
    """
    test_code = cleandoc("""
                    import pandas as pd
                    from sklearn.preprocessing import KBinsDiscretizer
                    import numpy as np
    
                    df = pd.DataFrame({'A': [1, 2, 10, 5]})
                    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
                    encoded_data = discretizer.fit_transform(df)
                    test_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                    encoded_data = discretizer.transform(test_df)
                    expected = np.array([[0.], [0.], [2.], [1.]])
                    assert np.allclose(encoded_data, expected)
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]

    expected_fit_transform = DagNode(1,
                                     BasicCodeLocation("<string-source>", 6),
                                     OperatorContext(OperatorType.TRANSFORMER,
                                                     FunctionInfo('sklearn.preprocessing._discretization',
                                                                  'KBinsDiscretizer')),
                                     DagNodeDetails('K-Bins Discretizer: fit_transform', ['array']),
                                     OptionalCodeInfo(CodeReference(6, 14, 6, 78),
                                                      "KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')"))
    expected_transform = DagNode(3,
                                 BasicCodeLocation("<string-source>", 6),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('sklearn.preprocessing._discretization',
                                                              'KBinsDiscretizer')),
                                 DagNodeDetails('K-Bins Discretizer: transform', ['array']),
                                 OptionalCodeInfo(CodeReference(6, 14, 6, 78),
                                                  "KBinsDiscretizer(n_bins=3, encode='ordinal', "
                                                  "strategy='uniform')"))

    compare(fit_transform_node, expected_fit_transform)
    compare(transform_node, expected_transform)

    expected_args = {'n_bins': 3, 'encode': 'ordinal', 'strategy': 'uniform'}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_fit_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)


def test_arg_capturing_one_hot_encoder():
    """
    Tests whether ArgumentCapturing works for the sklearn OneHotEncoder
    """
    test_code = cleandoc("""
                    import pandas as pd
                    from sklearn.preprocessing import label_binarize, OneHotEncoder
                    import numpy as np
    
                    df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                    one_hot_encoder = OneHotEncoder(sparse=False)
                    encoded_data = one_hot_encoder.fit_transform(df)
                    expected = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
                    print(encoded_data)
                    assert np.allclose(encoded_data, expected)
                    test_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                    encoded_data = one_hot_encoder.transform(test_df)
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]

    expected_fit_transform = DagNode(1,
                                     BasicCodeLocation("<string-source>", 6),
                                     OperatorContext(OperatorType.TRANSFORMER,
                                                     FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                     DagNodeDetails('One-Hot Encoder: fit_transform', ['array']),
                                     OptionalCodeInfo(CodeReference(6, 18, 6, 45), 'OneHotEncoder(sparse=False)'))
    expected_transform = DagNode(3,
                                 BasicCodeLocation("<string-source>", 6),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('sklearn.preprocessing._encoders',
                                                              'OneHotEncoder')),
                                 DagNodeDetails('One-Hot Encoder: transform', ['array']),
                                 OptionalCodeInfo(CodeReference(6, 18, 6, 45), 'OneHotEncoder(sparse=False)'))

    compare(fit_transform_node, expected_fit_transform)
    compare(transform_node, expected_transform)

    expected_args = {'categories': 'auto', 'drop': None, 'sparse': False, 'dtype': numpy.float64,
                     'handle_unknown': 'error'}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_fit_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)


def test_arg_capturing_simple_imputer():
    """
    Tests whether ArgumentCapturing works for the sklearn SimpleImputer
    """
    test_code = cleandoc("""
                    import pandas as pd
                    from sklearn.impute import SimpleImputer
                    import numpy as np
    
                    df = pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})
                    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                    imputed_data = imputer.fit_transform(df)
                    test_df = pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})
                    imputed_data = imputer.transform(test_df)
                    expected = np.array([['cat_a'], ['cat_a'], ['cat_a'], ['cat_c']])
                    assert np.array_equal(imputed_data, expected)
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]

    expected_fit_transform = DagNode(1,
                                     BasicCodeLocation("<string-source>", 6),
                                     OperatorContext(OperatorType.TRANSFORMER,
                                                     FunctionInfo('sklearn.impute._base', 'SimpleImputer')),
                                     DagNodeDetails('Simple Imputer: fit_transform', ['A']),
                                     OptionalCodeInfo(CodeReference(6, 10, 6, 72),
                                                      "SimpleImputer(missing_values=np.nan, strategy='most_frequent')"))
    expected_transform = DagNode(3,
                                 BasicCodeLocation("<string-source>", 6),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('sklearn.impute._base', 'SimpleImputer')),
                                 DagNodeDetails('Simple Imputer: transform', ['A']),
                                 OptionalCodeInfo(CodeReference(6, 10, 6, 72),
                                                  "SimpleImputer(missing_values=np.nan, strategy='most_frequent')"))

    compare(fit_transform_node, expected_fit_transform)
    compare(transform_node, expected_transform)

    expected_args = {'missing_values': numpy.nan, 'strategy': 'most_frequent', 'fill_value': None, 'verbose': 0,
                     'copy': True,
                     'add_indicator': False}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_fit_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)


def test_arg_capturing_function_transformer():
    """
    Tests whether ArgumentCapturing works for the sklearn FunctionTransformer
    """
    test_code = cleandoc("""
                    import pandas as pd
                    from sklearn.preprocessing import FunctionTransformer
                    import numpy as np
                    
                    def safe_log(x):
                        return np.log(x, out=np.zeros_like(x), where=(x!=0))
    
                    df = pd.DataFrame({'A': [1, 2, 10, 5]})
                    function_transformer = FunctionTransformer(lambda x: safe_log(x))
                    encoded_data = function_transformer.fit_transform(df)
                    test_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                    encoded_data = function_transformer.transform(test_df)
                    expected = np.array([[0.000000], [0.693147], [2.302585], [1.609438]])
                    assert np.allclose(encoded_data, expected)
                    """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        inspections=[ArgumentCapturing()])
    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]

    expected_fit_transform = DagNode(1,
                                     BasicCodeLocation("<string-source>", 9),
                                     OperatorContext(OperatorType.TRANSFORMER,
                                                     FunctionInfo('sklearn.preprocessing_function_transformer',
                                                                  'FunctionTransformer')),
                                     DagNodeDetails('Function Transformer: fit_transform', ['A']),
                                     OptionalCodeInfo(CodeReference(9, 23, 9, 65),
                                                      'FunctionTransformer(lambda x: safe_log(x))'))
    expected_transform = DagNode(3,
                                 BasicCodeLocation("<string-source>", 9),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('sklearn.preprocessing_function_transformer',
                                                              'FunctionTransformer')),
                                 DagNodeDetails('Function Transformer: transform', ['A']),
                                 OptionalCodeInfo(CodeReference(9, 23, 9, 65),
                                                  'FunctionTransformer(lambda x: safe_log(x))'))

    compare(fit_transform_node, expected_fit_transform)
    compare(transform_node, expected_transform)

    expected_args = {'validate': False, 'accept_sparse': False, 'check_inverse': True, 'kw_args': None,
                     'inv_kw_args': None}

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_fit_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)

    inspection_results_tree = inspector_result.dag_node_to_inspection_results[expected_transform]
    captured_args = inspection_results_tree[ArgumentCapturing()]
    compare(captured_args, expected_args)
