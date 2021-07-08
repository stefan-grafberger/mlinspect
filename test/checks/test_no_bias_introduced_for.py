"""
Tests whether NoMissingEmbeddings works
"""
import math
from inspect import cleandoc

from pandas import DataFrame
from testfixtures import compare

from mlinspect import DagNode, BasicCodeLocation, OperatorContext, OperatorType, FunctionInfo, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.checks import CheckStatus, NoBiasIntroducedFor, \
    NoBiasIntroducedForResult
from mlinspect.checks._no_bias_introduced_for import BiasDistributionChange
from mlinspect.instrumentation._dag_node import CodeReference


def test_no_bias_introduced_for_merge():
    """
    Tests whether RowLineage works for joins
    """
    test_code = cleandoc("""
            import pandas as pd

            df_a = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c', 'cat_b'], 'B': [1, 2, 4, 5, 7]})
            df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})
            df_merged = df_a.merge(df_b, on='B')
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_check(NoBiasIntroducedFor(['A'])) \
        .execute()

    check_result = inspector_result.check_to_check_results[NoBiasIntroducedFor(['A'])]
    expected_result = get_expected_check_result_merge()
    compare(check_result, expected_result)


def test_no_bias_introduced_simple_imputer():
    """
    Tests whether RowLineage works for joins
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
        .add_check(NoBiasIntroducedFor(['A'])) \
        .execute()

    check_result = inspector_result.check_to_check_results[NoBiasIntroducedFor(['A'])]
    expected_result = get_expected_check_result_simple_imputer()
    compare(check_result, expected_result)


def get_expected_check_result_merge():
    """ Expected result for the code snippet in test_no_bias_introduced_for_merge"""
    failing_dag_node = DagNode(2,
                               BasicCodeLocation('<string-source>', 5),
                               OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                               DagNodeDetails("on 'B'", ['A', 'B', 'C']),
                               OptionalCodeInfo(CodeReference(5, 12, 5, 36), "df_a.merge(df_b, on='B')"))

    change_df = DataFrame({'sensitive_column_value': ['cat_a', 'cat_b', 'cat_c'],
                           'count_before': [2, 2, 1],
                           'count_after': [2, 1, 1],
                           'ratio_before': [0.4, 0.4, 0.2],
                           'ratio_after': [0.5, 0.25, 0.25],
                           'relative_ratio_change': [(0.5 - 0.4) / 0.4, (.25 - 0.4) / 0.4, (0.25 - 0.2) / 0.2]})
    expected_distribution_change = BiasDistributionChange(failing_dag_node, False, (.25 - 0.4) / 0.4, change_df)
    expected_dag_node_to_change = {failing_dag_node: {'A': expected_distribution_change}}
    failure_message = 'A Join causes a min_relative_ratio_change of \'A\' by -0.37500000000000006, a value below the ' \
                      'configured minimum threshold -0.3!'
    expected_result = NoBiasIntroducedForResult(NoBiasIntroducedFor(['A']), CheckStatus.FAILURE, failure_message,
                                                expected_dag_node_to_change)
    return expected_result


def get_expected_check_result_simple_imputer():
    """ Expected result for the code snippet in test_no_bias_introduced_for_simple_imputer"""
    imputer_dag_node = DagNode(1,
                               BasicCodeLocation('<string-source>', 6),
                               OperatorContext(OperatorType.TRANSFORMER,
                                               FunctionInfo('sklearn.impute._base', 'SimpleImputer')),
                               DagNodeDetails('Simple Imputer: fit_transform', ['A']),
                               OptionalCodeInfo(CodeReference(6, 10, 6, 72),
                                                "SimpleImputer(missing_values=np.nan, strategy='most_frequent')"))

    change_df = DataFrame({'sensitive_column_value': ['cat_a', 'cat_c', math.nan],
                           'count_before': [2, 1, 1],
                           'count_after': [3, 1, 0],
                           'ratio_before': [0.5, 0.25, 0.25],
                           'ratio_after': [0.75, 0.25, 0.],
                           'relative_ratio_change': [0.5, 0., -1.]})
    expected_distribution_change = BiasDistributionChange(imputer_dag_node, True, 0., change_df)
    expected_dag_node_to_change = {imputer_dag_node: {'A': expected_distribution_change}}
    expected_result = NoBiasIntroducedForResult(NoBiasIntroducedFor(['A']), CheckStatus.SUCCESS, None,
                                                expected_dag_node_to_change)
    return expected_result
