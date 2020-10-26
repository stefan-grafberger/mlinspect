"""
The NoBiasIntroducedFor check
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, OrderedDict
import collections
from matplotlib import pyplot
from pandas import DataFrame

from mlinspect.checks._check import Check, CheckStatus, CheckResult
from mlinspect.inspections._histogram_for_columns import HistogramForColumns
from mlinspect.inspections._inspection import Inspection
from mlinspect.instrumentation._dag_node import OperatorType, DagNode
from mlinspect.inspections._inspection_result import InspectionResult


@dataclasses.dataclass(eq=True, frozen=True)
class BiasDistributionChange:
    """
    Did the histogram change too much for one given operation?
    """
    dag_node: DagNode
    acceptable_change: bool
    max_relative_change: float
    before_map: OrderedDict[str, int]
    after_map: OrderedDict[str, int]


@dataclasses.dataclass
class NoBiasIntroducedForResult(CheckResult):
    """
    Did the histogram change too much for some operations?
    """
    bias_distribution_change: OrderedDict[DagNode, OrderedDict[str, BiasDistributionChange]]


class NoBiasIntroducedFor(Check):
    """
    Does the user pipeline introduce bias because of operators like joins and selects?
    """

    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, sensitive_columns, max_relative_change=0.3):
        self.sensitive_columns = sensitive_columns
        self.max_relative_change = max_relative_change

    @property
    def check_id(self):
        """The id of the Check"""
        return tuple(self.sensitive_columns), self.max_relative_change

    @property
    def required_inspections(self) -> Iterable[Inspection]:
        """The inspections required for the check"""
        return [HistogramForColumns(self.sensitive_columns)]

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        dag = inspection_result.dag
        histograms = inspection_result.inspection_to_annotations[HistogramForColumns(self.sensitive_columns)]
        relevant_nodes = [node for node in dag.nodes if node.operator_type in {OperatorType.JOIN,
                                                                               OperatorType.SELECTION} or
                          (node.module == ('sklearn.impute._base', 'SimpleImputer', 'Pipeline') and
                           node.columns[0] in self.sensitive_columns)]
        check_status = CheckStatus.SUCCESS
        bias_distribution_change = collections.OrderedDict()
        issue_list = []
        for node in relevant_nodes:
            parents = list(dag.predecessors(node))
            column_results = collections.OrderedDict()
            for column in self.sensitive_columns:
                node_result = self.get_histograms_for_node_and_column(column, column_results, histograms, node,
                                                                      parents)
                if node_result == CheckStatus.FAILURE:
                    issue = "A {} significantly changes the '{}' distribution!".format(node.operator_type.value, column)
                    issue_list.append(issue)
                    check_status = CheckStatus.FAILURE

            bias_distribution_change[node] = column_results
        if issue_list:
            description = " ".join(issue_list)
        else:
            description = None
        return NoBiasIntroducedForResult(self, check_status, description, bias_distribution_change)

    def get_histograms_for_node_and_column(self, column, column_results, histograms, node, parents):
        """
        Compute histograms for a dag node like a join and a concrete sensitive column like race
        """
        # pylint: disable=too-many-locals, too-many-arguments
        after_map = histograms[node][column]
        before_map = collections.OrderedDict()
        for parent in parents:
            parent_histogram = histograms[parent][column]
            before_map = {**before_map, **parent_histogram}
        removed_groups = [group_key for group_key in after_map.keys() if group_key not in before_map and
                          group_key != "None"]
        if removed_groups:
            max_abs_change = 1.0
        else:
            before_count_all = sum(before_map.values())
            after_count_all = sum(after_map.values())
            abs_relative_changes = []
            for group_key in after_map:
                after_count = after_map[group_key]
                after_ratio = after_count / after_count_all
                before_count = before_map.get(group_key, 0)
                before_ratio = before_count / before_count_all or 0
                relative_change = (after_ratio - before_ratio) / after_ratio
                abs_relative_changes.append(abs(relative_change))
            max_abs_change = max(abs_relative_changes)
        all_changes_acceptable = max_abs_change <= self.max_relative_change
        if not all_changes_acceptable:
            node_column_result = CheckStatus.FAILURE
        else:
            node_column_result = CheckStatus.SUCCESS
        column_results[column] = BiasDistributionChange(node, all_changes_acceptable, max_abs_change,
                                                        before_map, after_map)
        return node_column_result

    @staticmethod
    def plot_distribution_change_histograms(distribution_change, filename=None, save_to_file=False):
        """
        Plot before and after histograms visualising a DistributionChange
        """
        pyplot.subplot(1, 2, 1)
        before_output_race_group = distribution_change.before_map

        # TODO: Move this into check
        dict_with_str_keys = {str(key): value for (key, value) in before_output_race_group.items()}
        sorted_items = sorted(dict_with_str_keys.items())
        before_output_race_group = collections.OrderedDict(sorted_items)

        keys = [str(key) for key in before_output_race_group.keys()]
        pyplot.bar(keys, before_output_race_group.values())
        pyplot.gca().set_title("before")
        pyplot.xticks(
            rotation=45,
            horizontalalignment='right',
        )

        pyplot.subplot(1, 2, 2)
        after_output_race_group = distribution_change.after_map

        # TODO: Move this into check
        dict_with_str_keys = {str(key): value for (key, value) in after_output_race_group.items()}
        sorted_items = sorted(dict_with_str_keys.items())
        after_output_race_group = collections.OrderedDict(sorted_items)

        keys = [str(key) for key in after_output_race_group.keys()]
        pyplot.bar(keys, after_output_race_group.values())
        pyplot.gca().set_title("after")
        pyplot.xticks(
            rotation=45,
            horizontalalignment='right',
        )

        fig = pyplot.gcf()
        fig.set_size_inches(12, 4)

        if save_to_file:
            fig.savefig(filename + '.svg', bbox_inches='tight')
            fig.savefig(filename + '.png', bbox_inches='tight', dpi=800)

        pyplot.show()
        pyplot.close()

    @staticmethod
    def get_distribution_change_as_df(distribution_change) -> DataFrame:
        """
        Get a pandas DataFrame with an overview of a DistributionChange
        """
        before_dict_with_str_keys = {str(key): value for (key, value) in distribution_change.before_map.items()}
        sorted_items = sorted(before_dict_with_str_keys.items())
        before_df = DataFrame(sorted_items, columns=["sensitive_column", "count_before"])

        after_dict_with_str_keys = {str(key): value for (key, value) in distribution_change.after_map.items()}
        sorted_items = sorted(after_dict_with_str_keys.items())
        after_df = DataFrame(sorted_items, columns=["sensitive_column", "count_after"])

        joined_df = before_df.merge(after_df, on="sensitive_column", how="outer")
        joined_df["absolute_change"] = joined_df["count_after"] - joined_df["count_before"]
        joined_df["relative_change"] = joined_df["count_after"] - joined_df["count_before"]  # TODO
        joined_df["acceptable_change"] = False  # TODO

        return joined_df

    @staticmethod
    def get_distribution_changes_overview_as_df(no_bias_check_result: NoBiasIntroducedForResult) -> DataFrame:
        """
        Get a pandas DataFrame with an overview of all DistributionChanges
        """
        operator_types = []
        code_references = []
        modules = []
        descriptions = []
        assert isinstance(no_bias_check_result.check, NoBiasIntroducedFor)
        sensitive_column_names = no_bias_check_result.check.sensitive_columns
        sensitive_column_names = ["'{}' distribution change above test threshold".format(name) for name in
                                  sensitive_column_names]
        sensitive_columns = []
        for _ in range(len(sensitive_column_names)):
            sensitive_columns.append([])
        for dag_node, distribution_change in no_bias_check_result.bias_distribution_change.items():
            operator_types.append(dag_node.operator_type)
            code_references.append(dag_node.code_reference)
            modules.append(dag_node.module)
            descriptions.append(dag_node.description)
            for index, change_info in enumerate(distribution_change.values()):
                sensitive_columns[index].append(not change_info.acceptable_change)
        return DataFrame(zip(operator_types, code_references, modules, descriptions, *sensitive_columns), columns=[
            "DagNode OperatorType",
            "DagNode CodeReference",
            "DagNode Monule",
            "DagNode Description",
            *sensitive_column_names
        ])
