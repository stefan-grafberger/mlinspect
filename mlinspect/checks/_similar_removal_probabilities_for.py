"""
The NoBiasIntroducedFor check
"""
import collections
import dataclasses
from typing import Iterable, Dict

from matplotlib import pyplot
from pandas import DataFrame

from mlinspect.checks._check import Check, CheckStatus, CheckResult
from mlinspect.inspections._histogram_for_columns import HistogramForColumns
from mlinspect.inspections._inspection import Inspection
from mlinspect.inspections._inspection_input import OperatorType, FunctionInfo
from mlinspect.inspections._inspection_result import InspectionResult
from mlinspect.instrumentation._dag_node import DagNode


@dataclasses.dataclass(eq=False, frozen=True)
class RemovalProbabilities:
    """
    Did the histogram change too much for one given operation?
    """
    dag_node: DagNode
    acceptable_probability_difference: bool
    max_probability_difference: float
    before_and_after_df: DataFrame

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.dag_node == other.dag_node and
                self.max_probability_difference == other.max_probability_difference and
                self.before_and_after_df.equals(other.before_and_after_df))


@dataclasses.dataclass
class SimilarRemovalProbabilitiesForResult(CheckResult):
    """
    Did the histogram change too much for some operations?
    """
    removal_probability_change: Dict[DagNode, Dict[str, RemovalProbabilities]]


class SimilarRemovalProbabilitiesFor(Check):
    """
    Does the user pipeline introduce bias because of operators like joins and selects? This check
    is based on computing removal probabilities. For all groups where values get removed, we expect the probability
    to be similar.
    """

    # pylint: disable=unnecessary-pass, too-few-public-methods

    def __init__(self, sensitive_columns, max_allowed_probability_difference=2.0):
        self.sensitive_columns = sensitive_columns
        self.max_allowed_probability_difference = max_allowed_probability_difference

    @property
    def check_id(self):
        """The id of the Check"""
        return tuple(self.sensitive_columns), self.max_allowed_probability_difference

    @property
    def required_inspections(self) -> Iterable[Inspection]:
        """The inspections required for the check"""
        return [HistogramForColumns(self.sensitive_columns)]

    def evaluate(self, inspection_result: InspectionResult) -> CheckResult:
        """Evaluate the check"""
        histograms, relevant_nodes = self.get_relevant_nodes_and_histograms(inspection_result)
        dag = inspection_result.dag
        check_status = CheckStatus.SUCCESS
        bias_distribution_change = collections.OrderedDict()
        issue_list = []
        for node in relevant_nodes:
            parents = list(dag.predecessors(node))
            column_results = collections.OrderedDict()
            for column in self.sensitive_columns:
                column_result = self.get_histograms_for_node_and_column(column, histograms, node, parents)
                column_results[column] = column_result
                if not column_result.acceptable_probability_difference:
                    issue = "A {} causes a max_probability_difference of '{}' by {}, a value above the " \
                            "configured maximum threshold {}!" \
                        .format(node.operator_info.operator.value, column, column_result.max_probability_difference,
                                self.max_allowed_probability_difference)
                    issue_list.append(issue)
                    check_status = CheckStatus.FAILURE

            bias_distribution_change[node] = column_results
        if issue_list:
            description = " ".join(issue_list)
        else:
            description = None
        return SimilarRemovalProbabilitiesForResult(self, check_status, description, bias_distribution_change)

    def get_relevant_nodes_and_histograms(self, inspection_result):
        """Get all DAG nodes relevant for this inspection and their histograms"""
        dag = inspection_result.dag
        histograms = {}
        for dag_node, inspection_results in inspection_result.dag_node_to_inspection_results.items():
            histograms[dag_node] = inspection_results[HistogramForColumns(self.sensitive_columns)]
        relevant_nodes = [node for node in dag.nodes if node.operator_info.operator in {OperatorType.JOIN,
                                                                                        OperatorType.SELECTION} or
                          (node.operator_info.function_info == FunctionInfo('sklearn.impute._base', 'SimpleImputer')
                           and set(node.details.columns).intersection(self.sensitive_columns))]
        return histograms, relevant_nodes

    def get_histograms_for_node_and_column(self, column, histograms, node, parents):
        """
        Compute histograms for a dag node like a join and a concrete sensitive column like race
        """
        # pylint: disable=too-many-locals, too-many-arguments
        after_map = histograms[node][column]
        after_df = DataFrame(after_map.items(), columns=["sensitive_column_value", "count_after"])

        before_map = {}
        for parent in parents:
            parent_histogram = histograms[parent][column]
            before_map = {**before_map, **parent_histogram}
        before_df = DataFrame(before_map.items(), columns=["sensitive_column_value", "count_before"])

        joined_df = before_df.merge(after_df, on="sensitive_column_value", how="outer")
        joined_df = joined_df.sort_values(by=['sensitive_column_value']).reset_index(drop=True)
        joined_df["count_before"] = joined_df["count_before"].fillna(0, downcast='infer')
        joined_df["count_after"] = joined_df["count_after"].fillna(0, downcast='infer')

        # Probability of removal
        joined_df["removed_records"] = joined_df["count_before"] - joined_df["count_after"]
        joined_df["removal_probability"] = joined_df["removed_records"] / joined_df["count_before"]
        # There might be classes where no records are being removed.
        # We should probably find a more principled method to do this at some point
        non_zero_probabilities = joined_df["removal_probability"] > 0.0
        removal_probability_min = joined_df[non_zero_probabilities]["removal_probability"].min()
        joined_df["normalized_removal_probability"] = joined_df["removal_probability"] / removal_probability_min
        joined_df.loc[joined_df['removed_records'] < 0, 'removal_probability'] = 0
        joined_df.loc[joined_df['removed_records'] < 0, 'normalized_removal_probability'] = 0

        # Dropping nan values (e.g., missing value imputation) is a distribution change we consider okay
        not_nan_columns = joined_df["sensitive_column_value"].notnull()
        not_nan_normalized_removal_probabilities = joined_df["normalized_removal_probability"].notnull()
        relevant_normalized_probabs = not_nan_columns & not_nan_normalized_removal_probabilities
        max_normalized_removal_probab = joined_df[relevant_normalized_probabs]["normalized_removal_probability"].max()
        acceptable_probability_difference = max_normalized_removal_probab <= self.max_allowed_probability_difference

        return RemovalProbabilities(node, acceptable_probability_difference, max_normalized_removal_probab, joined_df)

    @staticmethod
    def plot_distribution_change_histograms(distribution_change: RemovalProbabilities, filename=None,
                                            save_to_file=False):
        """
        Plot before and after histograms visualising a DistributionChange
        """
        pyplot.subplot(1, 2, 1)
        keys = distribution_change.before_and_after_df["sensitive_column_value"]
        keys = [str(key) for key in keys]  # Necessary because of null values
        before_values = distribution_change.before_and_after_df["count_before"]
        after_values = distribution_change.before_and_after_df["count_after"]

        pyplot.bar(keys, before_values)
        pyplot.gca().set_title("before")
        pyplot.xticks(
            rotation=45,
            horizontalalignment='right',
        )

        pyplot.subplot(1, 2, 2)

        pyplot.bar(keys, after_values)
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
    def plot_removal_probability_histograms(distribution_change: RemovalProbabilities, filename=None,
                                            save_to_file=False):
        """
        Plot before and after histograms visualising a DistributionChange
        """
        pyplot.subplot(1, 1, 1)
        keys = distribution_change.before_and_after_df["sensitive_column_value"]
        keys = [str(key) for key in keys]  # Necessary because of null values
        removal_probabilities = distribution_change.before_and_after_df["removal_probability"]

        pyplot.bar(keys, removal_probabilities)
        pyplot.gca().set_title("removal probability per member of sensitive group")
        pyplot.xticks(
            rotation=45,
            horizontalalignment='right',
        )

        fig = pyplot.gcf()
        fig.set_size_inches(6, 4)

        if save_to_file:
            fig.savefig(filename + '.svg', bbox_inches='tight')
            fig.savefig(filename + '.png', bbox_inches='tight', dpi=800)

        pyplot.show()
        pyplot.close()

    @staticmethod
    def get_removal_probabilities_overview_as_df(removal_probab_check_result: SimilarRemovalProbabilitiesForResult) \
            -> DataFrame:
        """
        Get a pandas DataFrame with an overview of all DistributionChanges
        """
        # pylint: disable=too-many-locals
        operator_types = []
        code_references = []
        function_infos = []
        code_snippets = []
        descriptions = []
        assert isinstance(removal_probab_check_result.check, SimilarRemovalProbabilitiesFor)
        sensitive_column_names = []
        for name in removal_probab_check_result.check.sensitive_columns:
            removal_probability_column_name = "'{}' probability difference below the configured maximum test " \
                                              "threshold".format(name)
            sensitive_column_names.append(removal_probability_column_name)

        sensitive_columns = []
        for _ in range(len(sensitive_column_names)):
            sensitive_columns.append([])
        for dag_node, removal_probability in removal_probab_check_result.removal_probability_change.items():
            operator_types.append(dag_node.operator_info.operator)
            if dag_node.optional_code_info is not None:
                code_references.append(dag_node.optional_code_info.code_reference)
                code_snippets.append(dag_node.optional_code_info.source_code)
            else:
                code_references.append(dag_node.code_location.lineno)
                code_snippets.append("You can enable code reference tracking for more details.")
            function_infos.append(dag_node.operator_info.function_info)

            descriptions.append(dag_node.details.description or "")
            for index, change_info in enumerate(removal_probability.values()):
                sensitive_columns[index].append(not change_info.acceptable_probability_difference)
        return DataFrame(zip(operator_types, descriptions, code_references, code_snippets, function_infos,
                             *sensitive_columns),
                         columns=[
                             "operator_type",
                             "description",
                             "code_reference",
                             "source_code",
                             "function_info",
                             *sensitive_column_names])
