"""
Tests whether ArgumentCapturing works
"""
from inspect import cleandoc

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
