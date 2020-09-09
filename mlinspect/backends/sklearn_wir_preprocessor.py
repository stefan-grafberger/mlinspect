"""
Preprocess Sklearn WIR nodes to enable DAG extraction
"""
import networkx
from more_itertools import pairwise

from ..instrumentation.wir_node import WirNode
from ..utils import traverse_graph_and_process_nodes, get_sorted_node_parents


class SklearnWirPreprocessor:
    """
    Preprocess Sklearn WIR nodes to enable DAG extraction
    """

    KNOWN_SINGLE_STEPS = {
        ('sklearn.preprocessing._encoders', 'OneHotEncoder'),
        ('sklearn.preprocessing._data', 'StandardScaler'),
        ('sklearn.tree._classes', 'DecisionTreeClassifier'),
        ('sklearn.impute._base', 'SimpleImputer'),
        ('sklearn.demo.healthcare.demo_utils', 'MyW2VTransformer'),
        ('sklearn.tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier')
    }

    KNOWN_MULTI_STEPS = {
        ('sklearn.compose._column_transformer', 'ColumnTransformer'),
        ('sklearn.pipeline', 'Pipeline'),
        ('sklearn.pipeline', 'fit')
    }

    def __init__(self):
        self.wir_node_to_sub_pipeline_start = {}
        self.wir_node_to_sub_pipeline_end = {}

    def preprocess_wir(self, graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Preprocess scikit-learn pipeline operations to hide the special pipeline
        declaration style from other parts of the library
        """

        def process_node(node, _):
            if node.module in self.KNOWN_SINGLE_STEPS:
                self.preprocess_single_step(node)
            elif node.module == ('sklearn.compose._column_transformer', 'ColumnTransformer'):
                self.preprocess_column_transformer(graph, node)
            if node.module == ('sklearn.pipeline', 'Pipeline'):
                self.preprocess_pipeline(graph, node)
            elif node.module == ('sklearn.pipeline', 'fit'):
                self.preprocess_pipeline_fit(graph, node)

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
        Preprocessing for ColumnTransformers: Introduce projection and concatenation nodes,
        create one transformer of the specified type for each column
        """
        transformers_arg = self.get_column_transformer_transformers_arg(graph, node)

        concat_module = (node.module[0], node.module[1], "Concatenation")
        concatenation_wir = WirNode(node.node_id, "Concatenation", "Call",
                                    node.code_reference, concat_module)

        self.wir_node_to_sub_pipeline_start[node] = []
        for transformer_tuple in transformers_arg:
            self.preprocess_column_transformer_transformer_tuple(concatenation_wir, graph, node, transformer_tuple)

        children = list(graph.successors(node))
        for child in children:
            graph.add_edge(concatenation_wir, child)

        self.wir_node_to_sub_pipeline_end[node] = concatenation_wir

    def preprocess_pipeline(self, graph, node):
        """
        Preprocessing for Pipelines: Chains the transformer nodes behind each other
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

    def preprocess_pipeline_fit(self, graph, node):
        """
        Preprocessing for Pipeline.fit: Inputs the train data into the beginning of the pipeline-chain,
        creates Train Data and Train Labels DAG nodes
        """
        actual_pipeline_node = self.get_pipeline_fit_pipeline_node(graph, node)
        data_node = self.get_pipeline_fit_arg_node(graph, node, 0)
        target_node_or_none = self.get_pipeline_fit_arg_node(graph, node, 1)

        pipeline_start = self.wir_node_to_sub_pipeline_start[actual_pipeline_node]
        pipeline_end = self.wir_node_to_sub_pipeline_end[actual_pipeline_node]

        new_fit_module = (node.module[0], node.module[1], "Pipeline")
        new_pipeline_fit_node = WirNode(node.node_id, node.name, node.operation,
                                        actual_pipeline_node.code_reference, new_fit_module)

        new_train_data_module = (node.module[0], node.module[1], "Train Data")
        new_pipeline_train_data_node = WirNode(node.node_id, "Train Data", node.operation,
                                               actual_pipeline_node.code_reference, new_train_data_module)
        graph.add_edge(data_node, new_pipeline_train_data_node)
        for start_node in pipeline_start:
            graph.add_edge(new_pipeline_train_data_node, start_node)

        graph.add_edge(pipeline_end, new_pipeline_fit_node)

        if target_node_or_none:
            new_train_data_module = (node.module[0], node.module[1], "Train Labels")
            new_pipeline_train_labels_node = WirNode(node.node_id, "Train Labels", node.operation,
                                                     actual_pipeline_node.code_reference, new_train_data_module)
            graph.add_edge(target_node_or_none, new_pipeline_train_labels_node)
            graph.add_edge(new_pipeline_train_labels_node, new_pipeline_fit_node)

    @staticmethod
    def get_column_transformer_transformers_arg(graph, node):
        """
        Get the 'transformers' list argument of ColumnTransformers
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
        Get the sub-pipelines specified in the 'steps' argument of Pipelines
        """
        # pylint: disable=too-many-locals
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
                new_transformer_module = (transformer.module[0],
                                          transformer.module[1], "Pipeline")

                new_transformer = WirNode(transformer.node_id, transformer.name, transformer.operation,
                                          transformer.code_reference, new_transformer_module,
                                          transformer.dag_operator_description)

                parents = list(graph.predecessors(transformer))
                for parent in parents:
                    graph.add_edge(parent, new_transformer)
                children = list(graph.successors(transformer))
                for child in children:
                    graph.add_edge(new_transformer, child)

                graph.remove_node(transformer)
                graph.add_node(new_transformer)

                self.wir_node_to_sub_pipeline_start[new_transformer] = [new_transformer]
                self.wir_node_to_sub_pipeline_end[new_transformer] = new_transformer
            else:
                new_transformer = transformer
            transformers_list.append(new_transformer)

        return transformers_list

    def get_pipeline_fit_pipeline_node(self, graph, node):
        """
        Get the 'pipeline' value of Pipeline.fit
        """
        direct_pipeline_parent_node = get_sorted_node_parents(graph, node)[0]
        actual_pipeline_node = self.get_sklearn_call_wir_node(graph, direct_pipeline_parent_node)

        assert actual_pipeline_node.operation == "Call"

        return actual_pipeline_node

    @staticmethod
    def get_pipeline_fit_arg_node(graph, node, index):
        """
        Get the train_data and train_label arguments of Pipeline.fit
        """
        parents = list(graph.predecessors(node))
        parents_with_arg_index = [(parent, graph.get_edge_data(parent, node)) for parent in parents]
        arg_node_list = [parent for parent in parents_with_arg_index if parent[1]['arg_index'] == index]

        arg_node = None
        if arg_node_list:
            arg_node = arg_node_list[0][0]

        return arg_node

    def preprocess_column_transformer_transformer_tuple(self, concatenation_wir, graph, node,
                                                        transformer_tuple):
        """
        Preprocessing for ColumnTransformers: Introduce projection nodes,
        create one transformer of the specified type for each column
        """
        # pylint: disable=too-many-locals
        sorted_tuple_parents = get_sorted_node_parents(graph, transformer_tuple)
        call_node = sorted_tuple_parents[1]
        column_list_node = sorted_tuple_parents[2]
        column_constant_nodes = list(graph.predecessors(column_list_node))
        projection_wirs = []
        for column_node in column_constant_nodes:
            projection_module = (node.module[0], node.module[1], "Projection")
            projection_description = "to {} (ColumnTransformer)".format([column_node.name])
            projection_wir = WirNode(column_node.node_id, column_node.name, node.operation,
                                     node.code_reference, projection_module, projection_description)
            projection_wirs.append(projection_wir)

            start_transformers, end_transformer = self.preprocess_column_transformer_copy_transformer_per_column(
                graph, call_node, column_node.node_id, column_node.name)

            parents = list(graph.predecessors(node))
            for parent in parents:
                graph.add_edge(parent, projection_wir)
            for start_transformer in start_transformers:
                graph.add_edge(projection_wir, start_transformer)
            graph.add_edge(end_transformer, concatenation_wir)
        self.wir_node_to_sub_pipeline_start[node].extend(projection_wirs)
        self.preprocess_column_transformer_delete_original_transformer(graph, call_node)

    def preprocess_column_transformer_copy_transformer_per_column(self, graph, transformer_node, new_node_id,
                                                                  description):
        """
        Preprocessing for ColumnTransformers: Each transformer in a ColumnTransformer needs to be copied for
        each column
        """
        transformer_node = self.get_sklearn_call_wir_node(graph, transformer_node)
        start_copy = list(self.wir_node_to_sub_pipeline_start[transformer_node])
        end_copy = self.wir_node_to_sub_pipeline_end[transformer_node]
        assert start_copy
        assert end_copy
        start_transformers = []
        end_transformer = []
        last_copied_wir = []

        def child_filter(child):
            return child.module and len(child.module) == 3

        def copy_node(current_node, _):
            new_module = (current_node.module[0], current_node.module[1], "Pipeline")
            new_description = "{}, Column: '{}'".format(current_node.dag_operator_description, description)
            copied_wir = WirNode(new_node_id, current_node.name, current_node.operation,
                                 current_node.code_reference, new_module, new_description)

            if current_node in start_copy:
                start_transformers.append(copied_wir)
                last_copied_wir.append(copied_wir)
            else:
                graph.add_edge(last_copied_wir[0], copied_wir)
                last_copied_wir[0] = copied_wir

            if current_node == end_copy:
                end_transformer.append(copied_wir)

        traverse_graph_and_process_nodes(graph, copy_node, list(start_copy), end_copy, child_filter)

        assert start_transformers
        assert end_transformer

        return start_transformers, end_transformer[0]

    def preprocess_column_transformer_delete_original_transformer(self, graph, transformer_node):
        """
        Preprocessing for ColumnTransformers: Each transformer in a ColumnTransformer needs to be copied for
        each column. Afterwards, the original nodes need to be deleted
        """
        transformer_node = self.get_sklearn_call_wir_node(graph, transformer_node)
        start_copy = list(self.wir_node_to_sub_pipeline_start[transformer_node])
        end_copy = self.wir_node_to_sub_pipeline_end[transformer_node]
        assert start_copy
        assert end_copy

        def child_filter(child):
            return child.module and len(child.module) == 3

        def copy_node(current_node, _):
            graph.remove_node(current_node)

        traverse_graph_and_process_nodes(graph, copy_node, list(start_copy), end_copy, child_filter)

    def get_sklearn_call_wir_node(self, graph, transformer):
        """
        Get a sklearn call that is the transformer WIR node or a parent of it.
        This is not straight-forward, as there can be many steps in-between, e.g., Assigns, Lists,
        Subscripts, Calls, etc.
        We currently deal with Assigns, but not the worst possible cases like
        nested lists or tuples with multiple transformers from which one is chosen with a
        subscript.
        """
        while transformer.operation == "Assign":
            transformer = list(graph.predecessors(transformer))[0]
        assert transformer.operation == "Call"
        assert transformer.module in self.KNOWN_SINGLE_STEPS.union(self.KNOWN_MULTI_STEPS)
        return transformer
