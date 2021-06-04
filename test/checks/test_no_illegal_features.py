"""
Tests whether NoMissingEmbeddings works
"""
from inspect import cleandoc

from testfixtures import compare, SequenceComparison, StringComparison

from mlinspect._pipeline_inspector import PipelineInspector
from mlinspect.checks import NoIllegalFeatures, CheckStatus, NoIllegalFeaturesResult


def test_no_illegal_features():
    """
    Tests whether NoIllegalFeatures works for joins
    """
    test_code = cleandoc("""
            import pandas as pd
            from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.tree import DecisionTreeClassifier

            data = pd.DataFrame({'age': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c'], 
                'C': ['cat_a', 'cat_b', 'cat_a', 'cat_c'], 'target': ['no', 'no', 'yes', 'yes']})
                
            column_transformer = ColumnTransformer(transformers=[
                ('numeric', StandardScaler(), ['age']),
                ('categorical', OneHotEncoder(sparse=False), ['B', 'C'])
            ])
            
            income_pipeline = Pipeline([
                ('features', column_transformer),
                ('classifier', DecisionTreeClassifier())])
            
            labels = label_binarize(data['target'], classes=['no', 'yes'])
            income_pipeline.fit(data, labels)
            """)

    inspector_result = PipelineInspector \
        .on_pipeline_from_string(test_code) \
        .add_check(NoIllegalFeatures(['C'])) \
        .execute()

    check_result = inspector_result.check_to_check_results[NoIllegalFeatures(['C'])]
    # pylint: disable=anomalous-backslash-in-string
    expected_result = NoIllegalFeaturesResult(NoIllegalFeatures(['C']), CheckStatus.FAILURE,
                                              StringComparison("Used illegal columns\: .*"),
                                              SequenceComparison('C', 'age', ordered=False))
    compare(check_result, expected_result)
