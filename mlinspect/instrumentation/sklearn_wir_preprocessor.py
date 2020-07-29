"""
Preprocess Sklearn WIR nodes to enable DAG extraction
"""
import networkx
from more_itertools import pairwise

from mlinspect.instrumentation.wir_vertex import WirVertex
from mlinspect.utils import traverse_graph_and_process_nodes, get_sorted_node_parents


class SklearnWirPreprocessor:
    """
    Preprocess Sklearn WIR nodes to enable DAG extraction
    """

    KNOWN_SINGLE_STEPS = {
        ('sklearn.preprocessing._encoders', 'OneHotEncoder'),
        ('sklearn.preprocessing._data', 'StandardScaler'),
        ('sklearn.tree._classes', 'DecisionTreeClassifier')
    }

    KNOWN_MULTI_STEPS = {
        ('sklearn.compose._column_transformer', 'ColumnTransformer'),
        ('sklearn.pipeline', 'Pipeline'),
        ('sklearn.pipeline', 'fit')
    }

    def __init__(self):
        self.wir_node_to_sub_pipeline_start = {}
        self.wir_node_to_sub_pipeline_end = {}

    def sklearn_wir_preprocessing(self, graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Re-orders scikit-learn pipeline operations in order to create a dag for them
        """

        def process_node(node, _):
            if node.module in self.KNOWN_SINGLE_STEPS:
                self.preprocess_single_step(node)
            elif node.module == ('sklearn.compose._column_transformer', 'ColumnTransformer'):
                self.preprocess_column_transformer(graph, node)
            if node.module == ('sklearn.pipeline', 'Pipeline'):
                self.preprocess_pipeline(graph, node)
            elif node.module == ('sklearn.pipeline', 'fit'):
                pass

        graph = traverse_graph_and_process_nodes(graph, process_node)
        return graph

    def preprocess_single_step(self, node):
        """
        Preprocessing for direct Transformer and Estimator calls
        """
        self.wir_node_to_sub_pipeline_start[node] = [node]
        self.wir_node_to_sub_pipeline_end[node] = node

    def preprocess_column_transformer(self, graph, node):
        """
        Re-orders scikit-learn ColumnTransformer operations in order to create a dag for them
        """
        transformers_arg = self.get_column_transformer_transformers_arg(graph, node)

        concat_module = (node.module[0], node.module[1], "Concatenation")
        concatenation_wir = WirVertex(node.node_id, "Concatenation", "Call", node.lineno,
                                      node.col_offset, concat_module)

        for transformer_tuple in transformers_arg:
            self.preprocess_column_transformer_transformer_tuple(concatenation_wir, graph, node, transformer_tuple)

        children = list(graph.successors(node))
        for child in children:
            graph.add_edge(concatenation_wir, child)

        self.wir_node_to_sub_pipeline_end[node] = concatenation_wir

    def preprocess_pipeline(self, graph, node):
        """
        Re-orders scikit-learn ColumnTransformer operations in order to create a dag for them
        """
        transformers_list = self.get_pipeline_steps_arg_transformers(graph, node)

        for (step, next_step) in pairwise(transformers_list):
            step_end = self.wir_node_to_sub_pipeline_end[step]
            next_step_start_nodes = self.wir_node_to_sub_pipeline_start[next_step]
            for next_step_start in next_step_start_nodes:
                graph.add_edge(step_end, next_step_start)

        pipeline_start = self.wir_node_to_sub_pipeline_start[transformers_list[0]]
        pipeline_end = self.wir_node_to_sub_pipeline_end[transformers_list[-1]]
        self.wir_node_to_sub_pipeline_start[node] = pipeline_start
        self.wir_node_to_sub_pipeline_end[node] = pipeline_end

    @staticmethod
    def get_column_transformer_transformers_arg(graph, node):
        """
        Get the 'transformers' argument of ColumnTransformers
        """
        parents = list(graph.predecessors(node))
        transformers_keyword_parent = [parent for parent in parents if parent.operation == "Keyword"
                                       and parent.name == "transformers"]
        if len(transformers_keyword_parent) != 0:
            transformer_keyword = transformers_keyword_parent[0]
            keyword_parents = list(graph.predecessors(transformer_keyword))
            transformers = keyword_parents[0]
        else:
            list_parents = [parent for parent in parents if parent.operation == "List"]
            transformers = list_parents[0]
        transformers_list = list(graph.predecessors(transformers))
        return transformers_list

    def get_pipeline_steps_arg_transformers(self, graph, node):
        """
        Get the 'transformers' argument of ColumnTransformers
        """
        parents = list(graph.predecessors(node))
        parents_with_arg_index = [(parent, graph.get_edge_data(parent, node)) for parent in parents]
        steps_list = [parent for parent in parents_with_arg_index if parent[1]['arg_index'] == 0][0][0]

        assert steps_list.operation == "List"

        steps_list_parents = get_sorted_node_parents(graph, steps_list)
        transformers_list = []
        for tuple_node in steps_list_parents:
            assert tuple_node.operation == "Tuple"
            tuple_parents = get_sorted_node_parents(graph, tuple_node)
            transformer = tuple_parents[1]
            transformer = self.get_sklearn_call_wir_node(graph, transformer)

            if transformer.module in self.KNOWN_SINGLE_STEPS:
                new_transformer_module = (transformer.module[0], transformer.module[1], "Pipeline")
                transformer = WirVertex(transformer.node_id, transformer.name, transformer.operation,
                                        transformer.lineno, transformer.col_offset, new_transformer_module)
                self.wir_node_to_sub_pipeline_start[transformer] = [transformer]
                self.wir_node_to_sub_pipeline_end[transformer] = transformer
            transformers_list.append(transformer)

        return transformers_list

    def preprocess_column_transformer_transformer_tuple(self, concatenation_wir, graph, node,
                                                        transformer_tuple):
        """
        Re-orders scikit-learn ColumnTransformer transformer tuple nodes in order to create a dag for them
        """
        # pylint: disable=too-many-locals
        sorted_tuple_parents = get_sorted_node_parents(graph, transformer_tuple)
        call_node = sorted_tuple_parents[1]
        column_list_node = sorted_tuple_parents[2]
        column_constant_nodes = list(graph.predecessors(column_list_node))
        projection_wirs = []
        for column_node in column_constant_nodes:
            projection_module = (node.module[0], node.module[1], "Projection")
            projection_wir = WirVertex(column_node.node_id, column_node.name, node.operation,
                                       node.lineno, node.col_offset, projection_module)
            projection_wirs.append(projection_wir)

            start_transformers, end_transformer = self.preprocess_column_transformer_copy_transformer_per_column(
                graph, call_node, column_node.node_id)

            parents = list(graph.predecessors(node))
            for parent in parents:
                graph.add_edge(parent, projection_wir)
            for start_transformer in start_transformers:
                graph.add_edge(projection_wir, start_transformer)
            graph.add_edge(end_transformer, concatenation_wir)
        self.wir_node_to_sub_pipeline_start[node] = projection_wirs

    def preprocess_column_transformer_copy_transformer_per_column(self, graph, transformer_node, new_node_id):
        """
        Each transformer in a ColumnTransformer needs to be copied for each column
        """
        transformer_node = self.get_sklearn_call_wir_node(graph, transformer_node)
        start_copy = set(self.wir_node_to_sub_pipeline_start[transformer_node])
        end_copy = self.wir_node_to_sub_pipeline_end[transformer_node]
        assert start_copy
        assert end_copy
        start_transformers = []
        end_transformer = []

        def copy_node(current_node, _):
            new_module = (current_node.module[0], current_node.module[1], "Pipeline")
            copied_wir = WirVertex(new_node_id, current_node.name, current_node.operation,
                                   current_node.lineno, current_node.col_offset, new_module)

            if current_node in start_copy:
                start_transformers.append(copied_wir)
            else:
                parents = list(graph.predecessors(current_node))
                relevant_parents = [parent for parent in parents if parent.node_id == new_node_id]
                for parent in relevant_parents:
                    graph.add_edge(parent, copied_wir)

            if current_node == end_copy:
                end_transformer.append(copied_wir)

        def child_filter(child):
            return child.module and len(child.module) == 3

        traverse_graph_and_process_nodes(graph, copy_node, list(start_copy), end_copy, child_filter)

        assert start_transformers
        assert end_transformer

        return start_transformers, end_transformer[0]

    def get_sklearn_call_wir_node(self, graph, transformer):
        """
        Get a sklearn call that is a parent to the transformer node.
        This is not straight-forward, as there can be steps in-between, e.g., Assigns.
        We currently deal with Assigns, but not the worst possible cases like
        nested lists or tuples with multiple transformers from which one is chosen with a
        subscript.
        """
        while transformer.operation == "Assign":
            transformer = list(graph.predecessors(transformer))[0]
        assert transformer.operation == "Call"
        assert transformer.module in self.KNOWN_SINGLE_STEPS.union(self.KNOWN_MULTI_STEPS)
        return transformer
