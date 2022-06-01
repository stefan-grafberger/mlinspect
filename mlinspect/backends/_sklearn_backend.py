"""
The scikit-learn backend
"""
from types import MappingProxyType
from typing import List, Dict

import numpy
import pandas
from scipy.sparse import csr_matrix

from ._backend import Backend, AnnotatedDfObject, BackendResult
from ._backend_utils import create_wrapper_with_annotations
from ._iter_creation import iter_input_annotation_output_sink_op, iter_input_annotation_output_nary_op
from ._pandas_backend import execute_inspection_visits_unary_operator, store_inspection_outputs, \
    execute_inspection_visits_data_source
from .. import OperatorType
from ..inspections import RowLineage
from ..instrumentation._pipeline_executor import singleton


class SklearnBackend(Backend):
    """
    The scikit-learn backend
    """

    @staticmethod
    def before_call(operator_context, input_infos: List[AnnotatedDfObject]):
        """The value or module a function may be called on"""
        if len(singleton.inspections) == 1 and isinstance(singleton.inspections[0], RowLineage) \
                and singleton.fast_lineage is True:
            SklearnBackend.lineage_only_before_call(input_infos, operator_context)
        else:
            if operator_context.operator == OperatorType.TRAIN_TEST_SPLIT:
                pandas_df = input_infos[0].result_data
                assert isinstance(pandas_df, pandas.DataFrame)
                pandas_df['mlinspect_index'] = range(0, len(pandas_df))
        return input_infos

    @staticmethod
    def lineage_only_before_call(input_infos, operator_context):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        if operator_context.operator == OperatorType.TRAIN_TEST_SPLIT:
            pandas_df = input_infos[0].result_data
            assert isinstance(pandas_df, pandas.DataFrame)
            lineage_inspection = singleton.inspections[0]
            inspection_name = str(lineage_inspection)
            input_infos[0].result_data['mlinspect_lineage'] = input_infos[0].result_annotation[inspection_name]

    @staticmethod
    def after_call(operator_context, input_infos: List[AnnotatedDfObject], return_value,
                   non_data_function_args: Dict[str, any] = MappingProxyType({})) \
            -> BackendResult:
        """The return value of some function"""
        if len(singleton.inspections) == 1 and isinstance(singleton.inspections[0], RowLineage) and \
                singleton.fast_lineage is True:
            return_value = SklearnBackend.lineage_only_after_call(input_infos, operator_context, return_value)
        else:
            if operator_context.operator == OperatorType.DATA_SOURCE:
                return_value = execute_inspection_visits_data_source(operator_context, return_value,
                                                                     non_data_function_args)
            elif operator_context.operator == OperatorType.TRAIN_TEST_SPLIT:
                train_data, test_data = return_value
                train_return_value = execute_inspection_visits_unary_operator(operator_context,
                                                                              input_infos[0].result_data,
                                                                              input_infos[0].result_annotation,
                                                                              train_data,
                                                                              True, non_data_function_args)
                test_return_value = execute_inspection_visits_unary_operator(operator_context,
                                                                             input_infos[0].result_data,
                                                                             input_infos[0].result_annotation,
                                                                             test_data,
                                                                             True, non_data_function_args)
                input_infos[0].result_data.drop("mlinspect_index", axis=1, inplace=True)
                train_data.drop("mlinspect_index", axis=1, inplace=True)
                test_data.drop("mlinspect_index", axis=1, inplace=True)
                return_value = BackendResult(train_return_value.annotated_dfobject,
                                             train_return_value.dag_node_annotation,
                                             test_return_value.annotated_dfobject,
                                             test_return_value.dag_node_annotation)
            elif operator_context.operator in {OperatorType.PROJECTION, OperatorType.PROJECTION_MODIFY,
                                               OperatorType.TRANSFORMER, OperatorType.TRAIN_DATA,
                                               OperatorType.TRAIN_LABELS,
                                               OperatorType.TEST_DATA, OperatorType.TEST_LABELS}:
                return_value = execute_inspection_visits_unary_operator(operator_context, input_infos[0].result_data,
                                                                        input_infos[0].result_annotation, return_value,
                                                                        False, non_data_function_args)
            elif operator_context.operator == OperatorType.ESTIMATOR:
                return_value = execute_inspection_visits_sink_op(operator_context,
                                                                 input_infos[0].result_data,
                                                                 input_infos[0].result_annotation,
                                                                 input_infos[1].result_data,
                                                                 input_infos[1].result_annotation,
                                                                 non_data_function_args)
            elif operator_context.operator == OperatorType.SCORE:
                return_value = execute_inspection_visits_nary_op(operator_context,
                                                                 input_infos,
                                                                 return_value,
                                                                 non_data_function_args)
            elif operator_context.operator == OperatorType.CONCATENATION:
                return_value = execute_inspection_visits_nary_op(operator_context, input_infos, return_value,
                                                                 non_data_function_args)
            else:
                raise NotImplementedError("SklearnBackend doesn't know any operations of type '{}' yet!"
                                          .format(operator_context.operator))
        return return_value

    @staticmethod
    def lineage_only_after_call(input_infos, operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        if operator_context.operator == OperatorType.DATA_SOURCE:
            # inspection annotation
            return_value = SklearnBackend.lineage_only_after_call_data_source(operator_context, return_value)
        elif operator_context.operator in {OperatorType.PROJECTION, OperatorType.PROJECTION_MODIFY,
                                           OperatorType.TRANSFORMER, OperatorType.TRAIN_DATA,
                                           OperatorType.TRAIN_LABELS, OperatorType.TEST_DATA,
                                           OperatorType.TEST_LABELS}:
            return_value = SklearnBackend.lineage_only_after_call_map(input_infos, operator_context, return_value)
        elif operator_context.operator in {OperatorType.TRAIN_TEST_SPLIT}:
            return_value = SklearnBackend.lineage_only_after_call_train_test_split(operator_context, return_value)
        elif operator_context.operator == OperatorType.ESTIMATOR:
            return_value = SklearnBackend.lineage_only_after_call_estimator(input_infos, operator_context, return_value)
        elif operator_context.operator == OperatorType.SCORE:
            return_value = SklearnBackend.lineage_only_after_call_score(input_infos, operator_context, return_value)
        elif operator_context.operator == OperatorType.CONCATENATION:
            return_value = SklearnBackend.lineage_only_after_call_concat(input_infos, operator_context, return_value)
        else:
            raise NotImplementedError("SklearnBackend doesn't know any operations of type '{}' yet!"
                                      .format(operator_context.operator))
        return return_value

    @staticmethod
    def lineage_only_after_call_concat(input_infos, operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        # inspection annotation
        lineage_inspection = singleton.inspections[0]
        inspection_name = str(lineage_inspection)
        annotations = [input_info.result_annotation for input_info in input_infos]
        annotations = [annotation_df.rename(columns={inspection_name: f'mlinspect_lineage_{index}'})
                       for index, annotation_df in enumerate(annotations)]
        annotations_df = pandas.concat(annotations, axis=1)
        if len(annotations) == 1:
            annotations_df['mlinspect_lineage'] = annotations_df['mlinspect_lineage_0']
        else:
            annotations_df['mlinspect_lineage'] = annotations_df['mlinspect_lineage_0'] + ';' + \
                                                  annotations_df['mlinspect_lineage_1']
            for index in range(1, len(annotations)):
                annotations_df['mlinspect_lineage'] = annotations_df['mlinspect_lineage'] + ';' + \
                                                      annotations_df[f'mlinspect_lineage_{index}']
            # annotations_df['mlinspect_lineage'] = annotations_df['mlinspect_lineage']\
            #     .apply(lambda value: ';'.join(set(value.split(';'))))
        for index in range(len(annotations)):
            annotations_df.drop(f'mlinspect_lineage_{index}', inplace=True, axis=1)
        inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        if materialize_for_this_operator:
            if isinstance(return_value, numpy.ndarray):
                pd_series = pandas.Series(list(return_value))
                pandas_return_value = pandas.DataFrame({'array': pd_series})
            elif isinstance(return_value, csr_matrix):
                pd_series = pandas.Series(list(return_value.toarray()))
                pandas_return_value = pandas.DataFrame({'array': pd_series})
            elif isinstance(return_value, pandas.DataFrame):
                pandas_return_value = return_value.reset_index(drop=True)
            else:
                assert False
            lineage_dag_annotation = pandas.concat([pandas_return_value, annotations_df], axis=1)
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                lineage_dag_annotation = lineage_dag_annotation.head(lineage_inspection.row_count)
        else:
            lineage_dag_annotation = None
        inspection_outputs[lineage_inspection] = lineage_dag_annotation
        # inspection output
        annotations_df = annotations_df.rename(columns={'mlinspect_lineage': inspection_name})
        return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
        return_value = BackendResult(return_value_with_annotation, inspection_outputs)
        return return_value

    @staticmethod
    def lineage_only_after_call_score(input_infos, operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        # inspection annotation
        lineage_inspection = singleton.inspections[0]
        inspection_name = str(lineage_inspection)
        annotations_df_data = input_infos[0].result_annotation
        annotations_df_data = annotations_df_data.rename(columns={inspection_name: 'mlinspect_lineage_x'})
        annotations_df_labels = input_infos[1].result_annotation
        annotations_df_labels = annotations_df_labels.rename(columns={inspection_name: 'mlinspect_lineage_y'})
        annotations_df = pandas.concat([annotations_df_data, annotations_df_labels], axis=1)
        annotations_df['mlinspect_lineage'] = annotations_df['mlinspect_lineage_x'] + ';' + \
                                              annotations_df['mlinspect_lineage_y']
        # annotations_df['mlinspect_lineage'] = annotations_df['mlinspect_lineage'] \
        #     .apply(lambda value: ';'.join(set(value.split(';'))))
        annotations_df.drop('mlinspect_lineage_x', inplace=True, axis=1)
        annotations_df.drop('mlinspect_lineage_y', inplace=True, axis=1)
        inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        if materialize_for_this_operator:
            if isinstance(return_value, numpy.ndarray):
                pd_series = pandas.Series(list(return_value))
                pandas_return_value = pandas.DataFrame({'array': pd_series})
            else:
                assert False
            lineage_dag_annotation = pandas.concat([pandas_return_value, annotations_df], axis=1)
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                lineage_dag_annotation = lineage_dag_annotation.head(lineage_inspection.row_count)
                lineage_dag_annotation = lineage_dag_annotation.rename(
                    columns={inspection_name: 'mlinspect_lineage'})
        else:
            lineage_dag_annotation = None
        inspection_outputs[lineage_inspection] = lineage_dag_annotation
        # inspection output
        return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
        return_value = BackendResult(return_value_with_annotation, inspection_outputs)
        return return_value

    @staticmethod
    def lineage_only_after_call_estimator(input_infos, operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        # inspection annotation
        lineage_inspection = singleton.inspections[0]
        inspection_name = str(lineage_inspection)
        annotations_df = input_infos[0].result_annotation
        inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        if materialize_for_this_operator:
            lineage_dag_annotation = annotations_df
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                lineage_dag_annotation = lineage_dag_annotation.head(lineage_inspection.row_count)
                lineage_dag_annotation = lineage_dag_annotation.rename(
                    columns={inspection_name: 'mlinspect_lineage'})
        else:
            lineage_dag_annotation = None
        inspection_outputs[lineage_inspection] = lineage_dag_annotation
        # inspection output
        return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
        return_value = BackendResult(return_value_with_annotation, inspection_outputs)
        return return_value

    @staticmethod
    def lineage_only_after_call_train_test_split(operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        # inspection annotation
        lineage_inspection = singleton.inspections[0]
        inspection_name = str(lineage_inspection)
        train_inspection_outputs = {}
        test_inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        train_data, test_data = return_value
        if materialize_for_this_operator:
            train_lineage_dag_annotation = train_data.reset_index(drop=True)
            test_lineage_dag_annotation = test_data.reset_index(drop=True)
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                train_lineage_dag_annotation = train_lineage_dag_annotation.head(lineage_inspection.row_count)
                test_lineage_dag_annotation = test_lineage_dag_annotation.head(lineage_inspection.row_count)
        else:
            train_lineage_dag_annotation = None
            test_lineage_dag_annotation = None
        train_inspection_outputs[lineage_inspection] = train_lineage_dag_annotation
        test_inspection_outputs[lineage_inspection] = test_lineage_dag_annotation
        # inspection annotation
        train_annotations_df = pandas.DataFrame(train_data.pop('mlinspect_lineage'))
        train_annotations_df = train_annotations_df.rename(columns={'mlinspect_lineage': inspection_name})
        test_annotations_df = pandas.DataFrame(test_data.pop('mlinspect_lineage'))
        test_annotations_df = test_annotations_df.rename(columns={'mlinspect_lineage': inspection_name})
        # inspection output
        train_return_value_data_with_annotation = create_wrapper_with_annotations(train_annotations_df,
                                                                                  train_data)
        test_return_value_data_with_annotation = create_wrapper_with_annotations(test_annotations_df,
                                                                                 test_data)
        return_value = BackendResult(train_return_value_data_with_annotation,
                                     train_inspection_outputs,
                                     test_return_value_data_with_annotation,
                                     test_inspection_outputs)
        return return_value

    @staticmethod
    def lineage_only_after_call_map(input_infos, operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        # inspection annotation
        lineage_inspection = singleton.inspections[0]
        inspection_name = str(lineage_inspection)
        annotations_df = input_infos[0].result_annotation
        inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        if materialize_for_this_operator:
            if isinstance(return_value, numpy.ndarray):
                pd_series = pandas.Series(list(return_value))
                pandas_return_value = pandas.DataFrame({'array': pd_series})
            elif isinstance(return_value, csr_matrix):
                pd_series = pandas.Series(list(return_value.toarray()))
                pandas_return_value = pandas.DataFrame({'array': pd_series})
            elif isinstance(return_value, pandas.DataFrame):
                pandas_return_value = return_value.reset_index(drop=True)
            elif isinstance(return_value, pandas.Series):
                pandas_return_value = pandas.DataFrame(return_value)
            else:
                assert False
            lineage_dag_annotation = pandas.concat([pandas_return_value, annotations_df], axis=1)
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                lineage_dag_annotation = lineage_dag_annotation.head(lineage_inspection.row_count)
                lineage_dag_annotation = lineage_dag_annotation.rename(
                    columns={inspection_name: 'mlinspect_lineage'})
        else:
            lineage_dag_annotation = None
        inspection_outputs[lineage_inspection] = lineage_dag_annotation
        # inspection output
        return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
        return_value = BackendResult(return_value_with_annotation, inspection_outputs)
        return return_value

    @staticmethod
    def lineage_only_after_call_data_source(operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        lineage_inspection = singleton.inspections[0]
        inspection_name = str(lineage_inspection)
        current_data_source = singleton.next_op_id - 1
        lineage_id_list_a = [f'({current_data_source},{row_id})' for row_id in range(len(return_value))]
        annotations_df = pandas.DataFrame({inspection_name: pandas.Series(lineage_id_list_a, dtype="object")})
        inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        if materialize_for_this_operator:
            pandas_return_value = pandas.DataFrame({'array': return_value})
            lineage_dag_annotation = pandas.concat([pandas_return_value, annotations_df], axis=1)
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                lineage_dag_annotation = lineage_dag_annotation.head(lineage_inspection.row_count)
            lineage_dag_annotation = lineage_dag_annotation.rename(
                columns={inspection_name: 'mlinspect_lineage'})
        else:
            lineage_dag_annotation = None
        inspection_outputs[lineage_inspection] = lineage_dag_annotation
        # inspection output
        return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
        return_value = BackendResult(return_value_with_annotation, inspection_outputs)
        return return_value


# -------------------------------------------------------
# Execute inspections functions
# -------------------------------------------------------

def execute_inspection_visits_sink_op(operator_context, data, data_annotation, target,
                                      target_annotation, non_data_function_args) -> BackendResult:
    """ Execute inspections """
    # pylint: disable=too-many-arguments
    inspection_count = len(singleton.inspections)
    iterators_for_inspections = iter_input_annotation_output_sink_op(inspection_count,
                                                                     data,
                                                                     data_annotation,
                                                                     target,
                                                                     target_annotation,
                                                                     operator_context,
                                                                     non_data_function_args)
    annotation_iterators = execute_visits(iterators_for_inspections)
    return_value = store_inspection_outputs(annotation_iterators, None)
    return return_value


def execute_inspection_visits_nary_op(operator_context, annotated_dfs: List[AnnotatedDfObject],
                                      return_value_df, non_data_function_args) -> BackendResult:
    """Execute inspections"""
    # pylint: disable=too-many-arguments
    inspection_count = len(singleton.inspections)
    iterators_for_inspections = iter_input_annotation_output_nary_op(inspection_count,
                                                                     annotated_dfs,
                                                                     return_value_df,
                                                                     operator_context,
                                                                     non_data_function_args)
    annotation_iterators = execute_visits(iterators_for_inspections)
    return_value = store_inspection_outputs(annotation_iterators, return_value_df)
    return return_value


def execute_visits(iterators_for_inspections):
    """
    After creating the iterators we need depending on the operator type, we need to execute the
    generic inspection visits
    """
    annotation_iterators = []
    for inspection_index, inspection in enumerate(singleton.inspections):
        iterator_for_inspection = iterators_for_inspections[inspection_index]
        annotations_iterator = inspection.visit_operator(iterator_for_inspection)
        annotation_iterators.append(annotations_iterator)
    return annotation_iterators
