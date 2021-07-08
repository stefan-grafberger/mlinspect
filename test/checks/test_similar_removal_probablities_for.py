"""
Tests whether NoMissingEmbeddings works
"""
import math
from inspect import cleandoc

import matplotlib
import pandas
from pandas import DataFrame
from testfixtures import compare

from mlinspect import DagNode, BasicCodeLocation, OperatorContext, OperatorType, FunctionInfo, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.checks import CheckStatus
from mlinspect.checks._similar_removal_probabilities_for import SimilarRemovalProbabilitiesFor, RemovalProbabilities, \
    SimilarRemovalProbabilitiesForResult
from mlinspect.instrumentation._dag_node import CodeReference


def test_removal_probab_for_merge():
    """
    Tests whether SimilarRemovalProbabilitiesFor works for joins
    """
    test_code = cleandoc("""
            import pandas as pd

            df_a = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c', 'cat_b'], 'B': [1, 2, 4, 5, 7]})
            df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})
            df_merged = df_a.merge(df_b, on='B')
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_check(SimilarRemovalProbabilitiesFor(['A'])) \
        .execute()

    check_result = inspector_result.check_to_check_results[SimilarRemovalProbabilitiesFor(['A'])]
    expected_result = get_expected_check_result_merge()
    compare(check_result, expected_result)


def test_removal_probab_simple_imputer():
    """
    Tests whether SimilarRemovalProbabilitiesFor works for imputation
    """
    test_code = cleandoc("""
            import pandas as pd
            from sklearn.impute import SimpleImputer
            import numpy as np

            df = pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            imputed_data = imputer.fit_transform(df)
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_check(SimilarRemovalProbabilitiesFor(['A'])) \
        .execute()

    check_result = inspector_result.check_to_check_results[SimilarRemovalProbabilitiesFor(['A'])]
    expected_result = get_expected_check_result_simple_imputer()
    compare(check_result, expected_result)


def test_removal_probab_dropna():
    """
    Tests whether SimilarRemovalProbabilitiesFor works for dropna
    """
    test_code = cleandoc("""
            import pandas as pd

            df = pd.DataFrame({'A': ['cat_a', 'cat_a', 'cat_c', 'cat_c', 'cat_c'], 
                               'B': [None, None, 1, 2, None]})
            df = df.dropna()
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_check(SimilarRemovalProbabilitiesFor(['A'])) \
        .execute()

    check_result = inspector_result.check_to_check_results[SimilarRemovalProbabilitiesFor(['A'])]
    expected_result = get_expected_check_result_dropna()
    compare(check_result, expected_result)

    overview = SimilarRemovalProbabilitiesFor.get_removal_probabilities_overview_as_df(check_result)
    expected_df = pandas.DataFrame({
        'operator_type': [OperatorType.SELECTION],
        'description': ['dropna'],
        'code_reference': [CodeReference(5, 5, 5, 16)],
        'source_code': ['df.dropna()'],
        'function_info': [FunctionInfo('pandas.core.frame', 'dropna')],
        "'A' probability difference below the configured maximum test threshold": [True]
    })
    pandas.testing.assert_frame_equal(overview, expected_df)
    matplotlib.use("template")  # Disable plt.show when executing nb as part of this test
    SimilarRemovalProbabilitiesFor.plot_removal_probability_histograms(
        list(check_result.removal_probability_change.values())[0]['A'])
    SimilarRemovalProbabilitiesFor.plot_distribution_change_histograms(
        list(check_result.removal_probability_change.values())[0]['A'])


def get_expected_check_result_merge():
    """ Expected result for the code snippet in test_no_bias_introduced_for_merge"""
    dag_node = DagNode(2,
                       BasicCodeLocation('<string-source>', 5),
                       OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                       DagNodeDetails("on 'B'", ['A', 'B', 'C']),
                       OptionalCodeInfo(CodeReference(5, 12, 5, 36), "df_a.merge(df_b, on='B')"))

    change_df = DataFrame({'sensitive_column_value': ['cat_a', 'cat_b', 'cat_c'],
                           'count_before': [2, 2, 1],
                           'count_after': [2, 1, 1],
                           'removed_records': [0, 1, 0],
                           'removal_probability': [0., 0.5, 0.],
                           'normalized_removal_probability': [0., 1., 0.]})
    expected_probabilities = RemovalProbabilities(dag_node, True, 1., change_df)
    expected_dag_node_to_change = {dag_node: {'A': expected_probabilities}}
    failure_message = None
    expected_result = SimilarRemovalProbabilitiesForResult(SimilarRemovalProbabilitiesFor(['A']), CheckStatus.SUCCESS,
                                                           failure_message, expected_dag_node_to_change)
    return expected_result


def get_expected_check_result_simple_imputer():
    """ Expected result for the code snippet in test_no_bias_introduced_for_simple_imputer"""
    dag_node = DagNode(1,
                       BasicCodeLocation('<string-source>', 6),
                       OperatorContext(OperatorType.TRANSFORMER,
                                       FunctionInfo('sklearn.impute._base', 'SimpleImputer')),
                       DagNodeDetails('Simple Imputer: fit_transform', ['A']),
                       OptionalCodeInfo(CodeReference(6, 10, 6, 72),
                                        "SimpleImputer(missing_values=np.nan, strategy='most_frequent')"))

    change_df = DataFrame({'sensitive_column_value': ['cat_a', 'cat_c', math.nan],
                           'count_before': [2, 1, 1],
                           'count_after': [3, 1, 0],
                           'removed_records': [-1, 0, 1],
                           'removal_probability': [0., 0., 1.],
                           'normalized_removal_probability': [0., 0., 1.]})
    expected_probabilities = RemovalProbabilities(dag_node, True, 0., change_df)
    expected_dag_node_to_change = {dag_node: {'A': expected_probabilities}}
    failure_message = None
    expected_result = SimilarRemovalProbabilitiesForResult(SimilarRemovalProbabilitiesFor(['A']), CheckStatus.SUCCESS,
                                                           failure_message, expected_dag_node_to_change)
    return expected_result


def get_expected_check_result_dropna():
    """ Expected result for the code snippet in test_no_bias_introduced_for_dropna"""
    dag_node = DagNode(1,
                       BasicCodeLocation('<string-source>', 5),
                       OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                       DagNodeDetails("dropna", ['A', 'B']),
                       OptionalCodeInfo(CodeReference(5, 5, 5, 16), "df.dropna()"))

    change_df = DataFrame({'sensitive_column_value': ['cat_a', 'cat_c'],
                           'count_before': [2, 3],
                           'count_after': [0, 2],
                           'removed_records': [2, 1],
                           'removal_probability': [1., 1. / 3.],
                           'normalized_removal_probability': [3., 1.]})
    expected_probabilities = RemovalProbabilities(dag_node, False, 3., change_df)
    expected_dag_node_to_change = {dag_node: {'A': expected_probabilities}}
    failure_message = "A Selection causes a max_probability_difference of 'A' by 3.0, a value above the configured " \
                      "maximum threshold 2.0!"
    expected_result = SimilarRemovalProbabilitiesForResult(SimilarRemovalProbabilitiesFor(['A']), CheckStatus.FAILURE,
                                                           failure_message, expected_dag_node_to_change)
    return expected_result
