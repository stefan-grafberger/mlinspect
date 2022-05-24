"""
The pandas backend
"""
from types import MappingProxyType
from typing import List, Dict

import pandas

from ._backend import Backend, AnnotatedDfObject, BackendResult
from ._backend_utils import build_annotation_df_from_iters, \
    create_wrapper_with_annotations
from ._iter_creation import iter_input_data_source, iter_input_annotation_output_resampled, \
    iter_input_annotation_output_map, iter_input_annotation_output_join
from .. import OperatorType, FunctionInfo
from ..inspections import RowLineage
from ..inspections._lineage import LineageId
from ..instrumentation._pipeline_executor import singleton


class PandasBackend(Backend):
    """
    The pandas backend
    """

    @staticmethod
    def before_call(operator_context, input_infos: List[AnnotatedDfObject]):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments
        if len(singleton.inspections) == 1 and isinstance(singleton.inspections[0], RowLineage) \
                and operator_context.operator \
                in {OperatorType.DATA_SOURCE, OperatorType.PROJECTION,
                    OperatorType.PROJECTION_MODIFY, OperatorType.SELECTION}:  # TODO: Add support for more operators
            if operator_context.operator == OperatorType.SELECTION:
                pandas_df = input_infos[0].result_data
                assert isinstance(pandas_df, pandas.DataFrame)
                lineage_inspection = singleton.inspections[0]
                inspection_name = str(lineage_inspection)
                input_infos[0].result_data['mlinspect_lineage'] = input_infos[0].result_annotation[inspection_name]
            elif operator_context.function_info == FunctionInfo('pandas.core.frame', 'merge'):
                raise NotImplementedError()
        else:
            if operator_context.operator == OperatorType.SELECTION:
                pandas_df = input_infos[0].result_data
                assert isinstance(pandas_df, pandas.DataFrame)
                pandas_df['mlinspect_index'] = range(0, len(pandas_df))
            elif operator_context.function_info == FunctionInfo('pandas.core.frame', 'merge'):
                first_pandas_df = input_infos[0].result_data
                assert isinstance(first_pandas_df, pandas.DataFrame)
                first_pandas_df['mlinspect_index_x'] = range(0, len(first_pandas_df))
                second_pandas_df = input_infos[1].result_data
                assert isinstance(second_pandas_df, pandas.DataFrame)
                second_pandas_df['mlinspect_index_y'] = range(0, len(second_pandas_df))
        return input_infos

    @staticmethod
    def after_call(operator_context, input_infos: List[AnnotatedDfObject], return_value,
                   non_data_function_args: Dict[str, any] = MappingProxyType({})) \
            -> BackendResult:
        """The return value of some function"""
        # pylint: disable=too-many-arguments,too-many-locals
        if len(singleton.inspections) == 1 and isinstance(singleton.inspections[0], RowLineage) \
                and operator_context.operator \
                in {OperatorType.DATA_SOURCE, OperatorType.PROJECTION,
                    OperatorType.PROJECTION_MODIFY, OperatorType.SELECTION,
                    OperatorType.GROUP_BY_AGG}:  # TODO: Add support for more operators
            print("optimized mode")
            if operator_context.operator in {OperatorType.DATA_SOURCE, OperatorType.GROUP_BY_AGG}:
                # inspection annotation
                lineage_inspection = singleton.inspections[0]
                inspection_name = str(lineage_inspection)
                current_data_source = singleton.data_source_count
                singleton.data_source_count += 1
                # TODO: Should we use a different format for performance reasons?
                # lineage_id_list_a = ["LineageId(0, " + str(row_id) + ")" for row_id in range(len(df_a))]
                lineage_id_list_a = [{LineageId(current_data_source, row_id)}
                                     for row_id in range(len(return_value))]
                annotations_df = pandas.DataFrame({inspection_name: pandas.Series(lineage_id_list_a, dtype="object")})
                inspection_outputs = {}
                materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                                (operator_context.operator
                                                 in lineage_inspection.operator_type_restriction)
                if materialize_for_this_operator:
                    if operator_context.operator == OperatorType.DATA_SOURCE:
                        reset_index_return_value = return_value.reset_index(drop=True)
                    else:
                        reset_index_return_value = return_value.reset_index(drop=False)
                    lineage_dag_annotation = pandas.concat([reset_index_return_value, annotations_df], axis=1)
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
            elif operator_context.operator in {OperatorType.PROJECTION, OperatorType.PROJECTION_MODIFY}:
                # inspection annotation
                lineage_inspection = singleton.inspections[0]
                inspection_name = str(lineage_inspection)
                # TODO: Should we use a different format for performance reasons?
                annotations_df = input_infos[0].result_annotation
                inspection_outputs = {}
                materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                                (operator_context.operator
                                                 in lineage_inspection.operator_type_restriction)
                if materialize_for_this_operator:
                    reset_index_return_value = return_value.reset_index(drop=True)
                    lineage_dag_annotation = pandas.concat([reset_index_return_value, annotations_df], axis=1)
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
            elif operator_context.operator in {OperatorType.SELECTION}:
                lineage_inspection = singleton.inspections[0]
                inspection_name = str(lineage_inspection)
                # TODO: Should we use a different format for performance reasons?
                inspection_outputs = {}
                materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                                (operator_context.operator
                                                 in lineage_inspection.operator_type_restriction)
                if materialize_for_this_operator:
                    reset_index_return_value = return_value.reset_index(drop=True)
                    lineage_dag_annotation = reset_index_return_value
                    if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                        lineage_dag_annotation = lineage_dag_annotation.head(lineage_inspection.row_count)
                else:
                    lineage_dag_annotation = None
                inspection_outputs[lineage_inspection] = lineage_dag_annotation
                # inspection annotation
                annotations_df = pandas.DataFrame(return_value.pop('mlinspect_lineage'))
                annotations_df = annotations_df.rename(columns={'mlinspect_lineage': inspection_name})
                # inspection output
                return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
                return_value = BackendResult(return_value_with_annotation, inspection_outputs)
            else:
                raise NotImplementedError()
        else:
            if operator_context.operator == OperatorType.DATA_SOURCE:
                return_value = execute_inspection_visits_data_source(operator_context, return_value,
                                                                     non_data_function_args)
            elif operator_context.operator == OperatorType.GROUP_BY_AGG:
                df_reset_index = return_value.reset_index()
                reset_index_return_value = execute_inspection_visits_data_source(operator_context, df_reset_index,
                                                                                 non_data_function_args)
                annotated_result_object = AnnotatedDfObject(return_value,
                                                            reset_index_return_value.annotated_dfobject.result_annotation)
                return_value = BackendResult(annotated_result_object, reset_index_return_value.dag_node_annotation)

            elif operator_context.operator == OperatorType.SELECTION:
                return_value = execute_inspection_visits_unary_operator(operator_context, input_infos[0].result_data,
                                                                        input_infos[0].result_annotation, return_value,
                                                                        True, non_data_function_args)
                input_infos[0].result_data.drop("mlinspect_index", axis=1, inplace=True)
            elif operator_context.operator in {OperatorType.PROJECTION, OperatorType.PROJECTION_MODIFY}:
                return_value = execute_inspection_visits_unary_operator(operator_context,
                                                                        input_infos[0].result_data,
                                                                        input_infos[0].result_annotation,
                                                                        return_value,
                                                                        False,
                                                                        non_data_function_args)
            elif operator_context.operator == OperatorType.JOIN:
                return_value = execute_inspection_visits_join(operator_context,
                                                              input_infos[0].result_data,
                                                              input_infos[0].result_annotation,
                                                              input_infos[1].result_data,
                                                              input_infos[1].result_annotation,
                                                              return_value,
                                                              non_data_function_args)
                input_infos[0].result_data.drop("mlinspect_index_x", axis=1, inplace=True)
                input_infos[1].result_data.drop("mlinspect_index_y", axis=1, inplace=True)

            else:
                raise NotImplementedError("PandasBackend doesn't know any operations of type '{}' yet!"
                                          .format(operator_context.operator))

        return return_value


# -------------------------------------------------------
# Execute inspections functions
# -------------------------------------------------------

def execute_inspection_visits_data_source(operator_context, return_value, non_data_function_args) -> BackendResult:
    """Execute inspections when the current operator is a data source and does not have parents in the DAG"""
    # pylint: disable=unused-argument
    inspection_count = len(singleton.inspections)
    iterators_for_inspections = iter_input_data_source(inspection_count, return_value, operator_context,
                                                       non_data_function_args)
    return_value = execute_visits_and_store_results(iterators_for_inspections, return_value)
    return return_value


def execute_inspection_visits_unary_operator(operator_context, input_data,
                                             input_annotations, return_value_df, resampled, non_data_function_args
                                             ) -> BackendResult:
    """Execute inspections when the current operator has one parent in the DAG"""
    # pylint: disable=too-many-arguments, unused-argument
    assert not resampled or "mlinspect_index" in return_value_df.columns
    inspection_count = len(singleton.inspections)
    if resampled:
        iterators_for_inspections = iter_input_annotation_output_resampled(inspection_count,
                                                                           input_data,
                                                                           input_annotations,
                                                                           return_value_df,
                                                                           operator_context,
                                                                           non_data_function_args)
    else:
        iterators_for_inspections = iter_input_annotation_output_map(inspection_count,
                                                                     input_data,
                                                                     input_annotations,
                                                                     return_value_df,
                                                                     operator_context,
                                                                     non_data_function_args)
    return_value = execute_visits_and_store_results(iterators_for_inspections, return_value_df)
    return return_value


def execute_inspection_visits_join(operator_context, input_data_one,
                                   input_annotations_one, input_data_two, input_annotations_two,
                                   return_value_df, non_data_function_args) -> BackendResult:
    """Execute inspections when the current operator has one parent in the DAG"""
    # pylint: disable=too-many-arguments, too-many-locals
    assert "mlinspect_index_x" in return_value_df
    assert "mlinspect_index_y" in return_value_df
    assert isinstance(input_data_one, pandas.DataFrame)
    assert isinstance(input_data_two, pandas.DataFrame)
    inspection_count = len(singleton.inspections)
    iterators_for_inspections = iter_input_annotation_output_join(inspection_count,
                                                                  input_data_one,
                                                                  input_annotations_one,
                                                                  input_data_two,
                                                                  input_annotations_two,
                                                                  return_value_df,
                                                                  operator_context,
                                                                  non_data_function_args)
    return_value = execute_visits_and_store_results(iterators_for_inspections, return_value_df)
    return return_value


def execute_visits_and_store_results(iterators_for_inspections, return_value) -> BackendResult:
    """
    After creating the iterators we need depending on the operator type, we need to execute the
    generic inspection visits and store the annotations in the resulting data frame
    """
    # pylint: disable=too-many-arguments
    annotation_iterators = []
    for inspection_index, inspection in enumerate(singleton.inspections):
        iterator_for_inspection = iterators_for_inspections[inspection_index]
        annotation_iterator = inspection.visit_operator(iterator_for_inspection)
        annotation_iterators.append(annotation_iterator)
    return_value = store_inspection_outputs(annotation_iterators, return_value)
    return return_value


# -------------------------------------------------------
# Store inspection results functions
# -------------------------------------------------------

def store_inspection_outputs(annotation_iterators, return_value) -> BackendResult:
    """
    Stores the inspection annotations for the rows in the dataframe and the
    inspection annotations for the DAG operators in a map
    """
    annotations_df = build_annotation_df_from_iters(singleton.inspections, annotation_iterators)
    inspection_outputs = {}
    for inspection in singleton.inspections:
        inspection_outputs[inspection] = inspection.get_operator_annotation_after_visit()
    new_return_value = create_wrapper_with_annotations(annotations_df, return_value)
    return BackendResult(new_return_value, inspection_outputs)
