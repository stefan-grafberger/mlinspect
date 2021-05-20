"""
The scikit-learn backend
"""
from typing import List

import pandas

from ._backend import Backend, AnnotatedDfObject
from ._pandas_backend import execute_inspection_visits_unary_operator
from ..instrumentation._dag_node import OperatorType


class SklearnBackend(Backend):
    """
    The scikit-learn backend
    """

    @staticmethod
    def before_call(function_info, operator_context, input_infos: List[AnnotatedDfObject]):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments
        if operator_context.operator == OperatorType.TRAIN_TEST_SPLIT:
            pandas_df = input_infos[0].result_data
            assert isinstance(pandas_df, pandas.DataFrame)
            pandas_df['mlinspect_index'] = range(0, len(pandas_df))
        return input_infos

    @staticmethod
    def after_call(function_info, operator_context, input_infos: List[AnnotatedDfObject], return_value) \
            -> AnnotatedDfObject:
        """The return value of some function"""
        # pylint: disable=too-many-arguments
        if operator_context.operator == OperatorType.TRAIN_TEST_SPLIT:
            return_value = execute_inspection_visits_unary_operator(operator_context,
                                                                    input_infos[0].result_data,
                                                                    input_infos[0].result_annotation,
                                                                    return_value,
                                                                    True)
            input_infos[0].result_data.drop("mlinspect_index", axis=1, inplace=True)
        elif operator_context.operator in {OperatorType.PROJECTION, OperatorType.PROJECTION_MODIFY}:
            return_value = execute_inspection_visits_unary_operator(operator_context,
                                                                    input_infos[0].result_data,
                                                                    input_infos[0].result_annotation,
                                                                    return_value,
                                                                    False)
        else:
            raise NotImplementedError("SklearnBackend doesn't know any operations of type '{}' yet!"
                                      .format(operator_context.operator))

        return return_value
