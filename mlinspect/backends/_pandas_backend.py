"""
The pandas backend
"""
from types import MappingProxyType
from typing import List, Dict

import numpy
import pandas

from ._backend import Backend, AnnotatedDfObject, BackendResult
from ._backend_utils import build_annotation_df_from_iters, \
    create_wrapper_with_annotations
from ._iter_creation import iter_input_data_source, iter_input_annotation_output_resampled, \
    iter_input_annotation_output_map, iter_input_annotation_output_join
from .. import OperatorType, FunctionInfo
from ..inspections import RowLineage
from ..instrumentation._pipeline_executor import singleton


class PandasBackend(Backend):
    """
    The pandas backend
    """

    @staticmethod
    def before_call(operator_context, input_infos: List[AnnotatedDfObject]):
        """The value or module a function may be called on"""
        if len(singleton.inspections) == 1 and isinstance(singleton.inspections[0], RowLineage) \
                and singleton.fast_lineage is True:
            PandasBackend.lineage_only_before_call(input_infos, operator_context)
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
    def lineage_only_before_call(input_infos, operator_context):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        # pylint: disable=too-many-locals
        if operator_context.operator == OperatorType.SELECTION:
            pandas_df = input_infos[0].result_data
            assert isinstance(pandas_df, pandas.DataFrame)
            annotations_df = input_infos[0].result_annotation
            input_infos[0].result_data[list(annotations_df.columns)] = annotations_df
        elif operator_context.function_info == FunctionInfo('pandas.core.frame', 'merge'):
            first_annotation_df = input_infos[0].result_annotation
            second_annotation_df = input_infos[1].result_annotation

            annotations_data_columns = set(first_annotation_df.columns)
            annotations_label_columns = set(second_annotation_df.columns)
            column_clashes = annotations_data_columns.intersection(annotations_label_columns)
            for column_clash in column_clashes:
                data_source, duplicate_index = column_clash.rsplit('_', 1)
                num_occurrences_in_data_columns = len([column for column in annotations_data_columns
                                                       if column.startswith(data_source)])
                new_duplicate_index = int(duplicate_index) + num_occurrences_in_data_columns
                new_col_name = f"{data_source}_{new_duplicate_index}"
                second_annotation_df = second_annotation_df.rename(columns={column_clash: new_col_name})

            first_pandas_df = input_infos[0].result_data
            assert isinstance(first_pandas_df, pandas.DataFrame)
            first_pandas_df[list(first_annotation_df.columns)] = first_annotation_df
            second_pandas_df = input_infos[1].result_data
            assert isinstance(second_pandas_df, pandas.DataFrame)
            second_pandas_df[list(second_annotation_df.columns)] = second_annotation_df

    @staticmethod
    def after_call(operator_context, input_infos: List[AnnotatedDfObject], return_value,
                   non_data_function_args: Dict[str, any] = MappingProxyType({})) \
            -> BackendResult:
        """The return value of some function"""
        if len(singleton.inspections) == 1 and isinstance(singleton.inspections[0], RowLineage) \
                and singleton.fast_lineage is True:
            return_value = PandasBackend.lineage_only_after_call(input_infos, non_data_function_args, operator_context,
                                                                 return_value)
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

    @staticmethod
    def lineage_only_after_call(input_infos, non_data_function_args, operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        if operator_context.operator in {OperatorType.DATA_SOURCE, OperatorType.GROUP_BY_AGG}:
            return_value = PandasBackend.lineage_only_after_call_data_source_groupby_agg(operator_context, return_value)
        elif operator_context.operator in {OperatorType.PROJECTION, OperatorType.PROJECTION_MODIFY}:
            return_value = PandasBackend.lineage_only_after_call_map(input_infos, operator_context, return_value)
        elif operator_context.operator in {OperatorType.SELECTION}:
            return_value = PandasBackend.lineage_only_after_call_filter(operator_context, return_value)
        elif operator_context.operator == OperatorType.JOIN:
            return_value = PandasBackend.lineage_only_after_call_join(non_data_function_args, operator_context,
                                                                      return_value)
        else:
            raise NotImplementedError()
        return return_value

    @staticmethod
    def lineage_only_after_call_join(non_data_function_args, operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        lineage_inspection = singleton.inspections[0]
        inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        # return_value['mlinspect_order_index'] = range(len(return_value))
        # singleton.con.register('lineage_df', return_value)
        # lineage_column = singleton.con.execute("""
        #     WITH unnested AS (
        #         SELECT DISTINCT mlinspect_order_index,
        #         UNNEST(str_split(mlinspect_lineage, ';')) AS mlinspect_lineage
        #         FROM lineage_df
        #     ),
        #     aggregated AS (
        #         SELECT mlinspect_order_index,
        #             string_agg(CAST(mlinspect_lineage AS string), ';') AS mlinspect_lineage
        #         FROM unnested
        #         GROUP BY mlinspect_order_index
        #     )
        #     SELECT mlinspect_lineage
        #     FROM aggregated
        #     ORDER BY mlinspect_order_index
        #     """).fetchdf()
        # return_value['mlinspect_lineage'] = lineage_column
        # return_value.drop('mlinspect_order_index', inplace=True, axis=1)
        # return_value['mlinspect_lineage'] = return_value['mlinspect_lineage'] \
        #     .apply(lambda value: ';'.join(set(value.split(';'))))
        return_value = return_value.reset_index(drop=True)
        if materialize_for_this_operator:
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                lineage_dag_annotation = return_value.head(lineage_inspection.row_count)
            else:
                lineage_dag_annotation = pandas.DataFrame(return_value, copy=True)
        else:
            lineage_dag_annotation = None
        inspection_outputs[lineage_inspection] = lineage_dag_annotation
        # inspection annotation
        columns_to_drop = [column for column in list(return_value.columns) if column.startswith('mlinspect_lineage')]
        annotations_df = pandas.DataFrame(return_value[columns_to_drop])
        return_value.drop(columns_to_drop, axis=1, inplace=True)
        # inspection output
        return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
        return_value = BackendResult(return_value_with_annotation, inspection_outputs)
        return return_value

    @staticmethod
    def lineage_only_after_call_filter(operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        lineage_inspection = singleton.inspections[0]
        inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        return_value = return_value.reset_index(drop=True)
        if materialize_for_this_operator:
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                lineage_dag_annotation = return_value.head(lineage_inspection.row_count)
            else:
                lineage_dag_annotation = pandas.DataFrame(return_value, copy=True)
        else:
            lineage_dag_annotation = None
        inspection_outputs[lineage_inspection] = lineage_dag_annotation
        # inspection annotation
        columns_to_drop = [column for column in list(return_value.columns) if column.startswith('mlinspect_lineage')]
        # TODO: Maybe a .reset_index(drop=True) necessary?
        annotations_df = pandas.DataFrame(return_value[columns_to_drop])
        return_value.drop(columns_to_drop, axis=1, inplace=True)
        # inspection output
        return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
        return_value = BackendResult(return_value_with_annotation, inspection_outputs)
        return return_value

    @staticmethod
    def lineage_only_after_call_map(input_infos, operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        # inspection annotation
        assert len(singleton.inspections) == 1
        lineage_inspection = singleton.inspections[0]
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
        else:
            lineage_dag_annotation = None
        inspection_outputs[lineage_inspection] = lineage_dag_annotation
        # inspection output
        return_value_with_annotation = create_wrapper_with_annotations(annotations_df, return_value)
        return_value = BackendResult(return_value_with_annotation, inspection_outputs)
        return return_value

    @staticmethod
    def lineage_only_after_call_data_source_groupby_agg(operator_context, return_value):
        """
        Optimised lineage inspection handling if only the lineage inspection is used
        """
        # inspection annotation
        assert len(singleton.inspections) == 1
        lineage_inspection = singleton.inspections[0]
        current_data_source = singleton.next_op_id - 1
        lineage_column_name = f"mlinspect_lineage_{current_data_source}_0"
        annotations_df = pandas.DataFrame({lineage_column_name: range(len(return_value))})
        inspection_outputs = {}
        materialize_for_this_operator = (lineage_inspection.operator_type_restriction is None) or \
                                        (operator_context.operator
                                         in lineage_inspection.operator_type_restriction)
        if isinstance(return_value, (pandas.DataFrame, pandas.Series)) \
                and operator_context.operator == OperatorType.DATA_SOURCE:
            return_value.reset_index(drop=True, inplace=True)
            data_df = return_value
        elif isinstance(return_value, (pandas.DataFrame, pandas.Series)):
            return_value.reset_index(drop=False, inplace=True)
            data_df = return_value
        else:
            data_df = pandas.DataFrame({'array': return_value})
        if materialize_for_this_operator:
            lineage_dag_annotation = pandas.concat([data_df, annotations_df], axis=1)
            if lineage_inspection.row_count != RowLineage.ALL_ROWS:
                lineage_dag_annotation = lineage_dag_annotation.head(lineage_inspection.row_count)
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
