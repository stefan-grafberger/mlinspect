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
        self.wir_node_to_sub_pipeline_beginning = {}
        self.wir_node_to_sub_pipeline_end = {}
        self.wir_nodes_to_delete_when_done = set()

    # create a map that maps from pipeline entity to list of start ast nodes and an end ast node
    # add processing for scaler and onehot encoder etc too to initialize the map. then update it in
    # column transformer. fit and pipeline can then use this map too

    def sklearn_wir_preprocessing(self, graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Re-orders scikit-learn pipeline operations in order to create a dag for them
        """

        def process_node(node, _):
            if node.module in self.KNOWN_SINGLE_STEPS:
                self.wir_node_to_sub_pipeline_beginning[node] = [node]
                self.wir_node_to_sub_pipeline_end[node] = node
            elif node.module == ('sklearn.compose._column_transformer', 'ColumnTransformer'):
                self.preprocess_column_transformer(graph, node)
            if node.module == ('sklearn.pipeline', 'Pipeline'):
                pass
            elif node.module == ('sklearn.pipeline', 'fit'):
                pass

        graph.remove_nodes_from(self.wir_nodes_to_delete_when_done)
        return traverse_graph_and_process_nodes(graph, process_node)

    def preprocess_column_transformer(self, graph, node):
        """
        Re-orders scikit-learn ColumnTransformer operations in order to create a dag for them
        """
        parents = list(graph.predecessors(node))
        children = list(graph.successors(node))

        transformers_list = SklearnWirPreprocessor.get_column_transformer_transformers_list(graph, parents)

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
        self.wir_nodes_to_delete_when_done.add(node)

    def preprocess_column_transformer_transformer_tuple(self, concatenation_wir, graph, node,
                                                        transformer_tuple):
        """
        Re-orders scikit-learn ColumnTransformer transformer tuple nodes in order to create a dag for them
        """
        # pylint: disable=too-many-locals
        sorted_tuple_parents = SklearnWirPreprocessor.get_sorted_node_parents(graph, transformer_tuple)
        call_node = sorted_tuple_parents[1]
        column_list_node = sorted_tuple_parents[2]
        column_constant_nodes = list(graph.predecessors(column_list_node))
        projection_wirs = []
        for column_node in column_constant_nodes:
            projection_module = (node.module[0], node.module[1], "Projection")
            projection_wir = WirVertex(column_node.node_id, column_node.name, node.operation,
                                       node.lineno, node.col_offset, projection_module)
            projection_wirs.append(projection_wir)

            # FIXME here i need a deep copy
            # FIXME and we can not assume that it is only a single node. find pipeline beginning and end with the map
            new_call_module = (call_node.module[0], call_node.module[1], "Transformer")
            new_call_wir = WirVertex(column_node.node_id, call_node.name, call_node.operation,
                                     call_node.lineno, call_node.col_offset, new_call_module)
            self.wir_nodes_to_delete_when_done.add(call_node) # plus other stuff
            start_transformers = [new_call_wir]
            end_transformer = new_call_wir

            # end

            parents = list(graph.predecessors(node))
            for parent in parents:
                graph.add_edge(parent, projection_wir)
            for start_transformer in start_transformers:
                graph.add_edge(projection_wir, start_transformer)
            graph.add_edge(end_transformer, concatenation_wir)
        self.wir_node_to_sub_pipeline_beginning[node] = projection_wirs
        graph.remove_nodes_from(sorted_tuple_parents)
        graph.remove_node(transformer_tuple)

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
