"""
Preprocess Sklearn WIR nodes to enable DAG extraction
"""
import networkx

from mlinspect.instrumentation.wir_vertex import WirVertex
from mlinspect.utils import traverse_graph_and_process_nodes


class SklearnWirPreprocessor:
    """
    Preprocess Sklearn WIR nodes to enable DAG extraction
    """

    KNOWN_SINGLE_STEPS = {
        ('sklearn.preprocessing._encoders', 'OneHotEncoder'),
        ('sklearn.preprocessing._data', 'StandardScaler')
    }

    def __init__(self):
        self.wir_node_to_sub_pipeline_start = {}
        self.wir_node_to_sub_pipeline_end = {}

    # create a map that maps from pipeline entity to list of start ast nodes and an end ast node
    # add processing for scaler and onehot encoder etc too to initialize the map. then update it in
    # column transformer. fit and pipeline can then use this map too

    def sklearn_wir_preprocessing(self, graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Re-orders scikit-learn pipeline operations in order to create a dag for them
        """

        def process_node(node, _):
            if node.module in self.KNOWN_SINGLE_STEPS:
                self.wir_node_to_sub_pipeline_start[node] = [node]
                self.wir_node_to_sub_pipeline_end[node] = node
            elif node.module == ('sklearn.compose._column_transformer', 'ColumnTransformer'):
                self.preprocess_column_transformer(graph, node)
            if node.module == ('sklearn.pipeline', 'Pipeline'):
                pass
            elif node.module == ('sklearn.pipeline', 'fit'):
                pass

        graph = traverse_graph_and_process_nodes(graph, process_node)
        return graph

    def preprocess_column_transformer(self, graph, node):
        """
        Re-orders scikit-learn ColumnTransformer operations in order to create a dag for them
        """
        parents = list(graph.predecessors(node))
        children = list(graph.successors(node))

        transformers_list = self.get_column_transformer_transformers_list(graph, parents)

        # Concatenation node
        concat_module = (node.module[0], node.module[1], "Concatenation")
        concatenation_wir = WirVertex(node.node_id, "Concatenation", "Call", node.lineno,
                                      node.col_offset, concat_module)
        for transformer_tuple in transformers_list:
            self.preprocess_column_transformer_transformer_tuple(concatenation_wir, graph, node, transformer_tuple)
        graph.remove_nodes_from(transformers_list)
        for child in children:
            graph.add_edge(concatenation_wir, child)

        self.wir_node_to_sub_pipeline_end[node] = concatenation_wir

    def preprocess_column_transformer_transformer_tuple(self, concatenation_wir, graph, node,
                                                        transformer_tuple):
        """
        Re-orders scikit-learn ColumnTransformer transformer tuple nodes in order to create a dag for them
        """
        # pylint: disable=too-many-locals
        sorted_tuple_parents = self.get_sorted_node_parents(graph, transformer_tuple)
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
                graph, call_node, column_node)

            # end

            parents = list(graph.predecessors(node))
            for parent in parents:
                graph.add_edge(parent, projection_wir)
            for start_transformer in start_transformers:
                graph.add_edge(projection_wir, start_transformer)
            graph.add_edge(end_transformer, concatenation_wir)
        self.wir_node_to_sub_pipeline_start[node] = projection_wirs

    def preprocess_column_transformer_copy_transformer_per_column(self, graph, transformer_node, column_node):
        """
        Each transformer in a ColumnTransformer needs to be copied for each column
        """
        start_copy = set(self.wir_node_to_sub_pipeline_start[transformer_node])
        end_copy = self.wir_node_to_sub_pipeline_end[transformer_node]
        assert start_copy
        assert end_copy
        start_transformers = []
        end_transformer = []

        def copy_node(current_node, _):
            new_call_module = (current_node.module[0], current_node.module[1], "Transformer")
            copied_wir = WirVertex(column_node.node_id, current_node.name, current_node.operation,
                                   current_node.lineno, current_node.col_offset, new_call_module)

            if current_node in start_copy:
                start_transformers.append(copied_wir)
            else:
                parents = list(graph.predecessors(current_node))
                relevant_parents = [parent for parent in parents if parent.node_id == column_node.node_id]
                for parent in relevant_parents:
                    graph.add_edge(parent, copied_wir)

            if current_node == end_copy:
                end_transformer.append(current_node)

        self.traverse_graph_and_process_nodes_with_start_and_end(graph, list(start_copy), end_copy, copy_node)

        assert start_transformers
        assert end_transformer

        return start_transformers, end_transformer[0]

    @staticmethod
    def traverse_graph_and_process_nodes_with_start_and_end(graph: networkx.DiGraph, start_nodes, end_node, func):
        """
        Traverse the WIR node by node from top to bottom
        """
        current_nodes = start_nodes
        processed_nodes = set()
        while len(current_nodes) != 0:
            node = current_nodes.pop(0)
            processed_nodes.add(node)
            children = list(graph.successors(node))
            relevant_children = [child for child in children if child.module and len(child.module) == 3]

            # Nodes can have multiple parents, only want to process them once we processed all parents
            if node != end_node:
                for child in relevant_children:
                    if child not in processed_nodes:
                        if processed_nodes.issuperset(graph.predecessors(child)):
                            current_nodes.append(child)

            func(node, processed_nodes)
        return graph

    @staticmethod
    def get_sorted_node_parents(graph, node_with_parents):
        """
        Get the parent nodes of e.g., a tuple sorted by argument index.
        """
        node_parents = list(graph.predecessors(node_with_parents))
        node_parents_with_arg_index = [(node_parent, graph.get_edge_data(node_parent, node_with_parents))
                                       for node_parent in node_parents]
        sorted_node_parents_with_arg_index = sorted(node_parents_with_arg_index, key=lambda x: x[1]['arg_index'])
        sorted_node_parents = [node_parent[0] for node_parent in sorted_node_parents_with_arg_index]
        return sorted_node_parents

    @staticmethod
    def get_column_transformer_transformers_list(graph, parents):
        """
        Get the 'transformers' argument of ColumnTransformers
        """
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
