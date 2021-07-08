"""
Tests whether NoMissingEmbeddings works
"""
from inspect import cleandoc

from testfixtures import compare

from demo.feature_overview.missing_embeddings import MissingEmbeddingsInfo
from demo.feature_overview.no_missing_embeddings import NoMissingEmbeddings, NoMissingEmbeddingsResult
from example_pipelines.healthcare import custom_monkeypatching
from mlinspect import DagNode, BasicCodeLocation, OperatorContext, OperatorType, FunctionInfo, DagNodeDetails, \
    OptionalCodeInfo
from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.checks import CheckStatus
from mlinspect.instrumentation._dag_node import CodeReference


def test_no_missing_embeddings():
    """
    Tests whether NoMissingEmbeddings works for joins
    """
    test_code = cleandoc("""
            import pandas as pd
            from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer

            df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
            word_to_vec = MyW2VTransformer(min_count=2, size=2, workers=1)
            encoded_data = word_to_vec.fit_transform(df)
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_check(NoMissingEmbeddings()) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .execute()

    check_result = inspector_result.check_to_check_results[NoMissingEmbeddings()]
    expected_failed_dag_node_with_result = {
        DagNode(1,
                BasicCodeLocation('<string-source>', 5),
                OperatorContext(OperatorType.TRANSFORMER,
                                FunctionInfo('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer')),
                DagNodeDetails('Word2Vec: fit_transform', ['array']),
                OptionalCodeInfo(CodeReference(5, 14, 5, 62), 'MyW2VTransformer(min_count=2, size=2, workers=1)'))
        : MissingEmbeddingsInfo(2, ['cat_b', 'cat_c'])}
    expected_result = NoMissingEmbeddingsResult(NoMissingEmbeddings(10), CheckStatus.FAILURE,
                                                'Missing embeddings were found!', expected_failed_dag_node_with_result)
    compare(check_result, expected_result)
