"""
Extract a DAG from the WIR (Workflow Intermediate Representation)
"""
import networkx

from mlinspect.instrumentation.dag_vertex import DagVertex
from mlinspect.instrumentation.wir_vertex import WirVertex


class WirToDagTransformer:
    """
    Extract DAG from the WIR (Workflow Intermediate Representation)
    """

    OPERATOR_MAP = {
        ('pandas.io.parsers', 'read_csv'): "Data Source",
        ('pandas.core.frame', 'dropna'): "Selection",
        ('pandas.core.frame', '__getitem__'): "Projection",
        ('sklearn.preprocessing._label', 'label_binarize'): "Projection (Modify)",
        #('sklearn.compose._column_transformer', 'ColumnTransformer'): "Projection",  # FIXME
        #('sklearn.preprocessing._encoders', 'OneHotEncoder'): "Transformer",
        #('sklearn.preprocessing._data', 'StandardScaler'): "Transformer",
        #('sklearn.tree._classes', 'DecisionTreeClassifier'): "Estimator",
        #('sklearn.pipeline', 'fit'): "Fit Transformers and Estimators"
    }

    @staticmethod
    def sklearn_wir_preprocessing(graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Re-orders scikit-learn pipeline operations in order to create a dag for them
        """
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
        processed_nodes = set()
        while len(current_nodes) != 0:
            node = current_nodes.pop(0)
            processed_nodes.add(node)
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))

            # Nodes can have multiple parents, only want to process them once we processed all parents
            for child in children:
                if child not in processed_nodes:
                    if processed_nodes.issuperset(graph.predecessors(child)):
                        current_nodes.append(child)

            if node.module == ('sklearn.pipeline', 'Pipeline'):
                pass
            elif node.module == ('sklearn.compose._column_transformer', 'ColumnTransformer'):
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
                module_name, func_name = node.module
                new_module_name = "mlinspect." + module_name
                new_module = (new_module_name, func_name)
                concatenation_wir = WirVertex(node.node_id, "Concatenation", node.operation, node.lineno,
                                              node.col_offset, new_module)
                for transformer_tuple in transformers_list:
                    tuple_parents = list(graph.predecessors(transformer_tuple))
                    tuple_parents_with_arg_index = [(tuple_parent, graph.get_edge_data(tuple_parent, transformer_tuple))
                                                    for tuple_parent in tuple_parents]
                    tuple_parents = sorted(tuple_parents_with_arg_index, key=lambda x: x[1]['arg_index'])
                    call_node = tuple_parents[1][0]
                    column_list_node = tuple_parents[2][0]
                    column_nodes = list(graph.predecessors(column_list_node))
                    for column_node in column_nodes:
                        column_name = column_node.name
                        projection_wir = WirVertex(node.node_id, "Projection", node.operation, node.lineno,
                                                   node.col_offset, new_module)

                        for parent in parents:
                            graph.add_edge(parent, projection_wir)
                        # transformer etc
                        graph.add_edge(projection_wir, call_node)
                        graph.add_edge(projection_wir, concatenation_wir)
                for child in children:
                    graph.add_edge(concatenation_wir, child)
                graph.remove_node(node)
            elif node.module == ('sklearn.pipeline', 'fit'):
                pass

        return graph

    @staticmethod
    def remove_all_nodes_but_calls_and_subscripts(graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Removes all nodes that can not be a operator we might care about
        """
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
        processed_nodes = set()
        while len(current_nodes) != 0:
            node = current_nodes.pop(0)
            processed_nodes.add(node)
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))

            # Nodes can have multiple parents, only want to process them once we processed all parents
            for child in children:
                if child not in processed_nodes:
                    if processed_nodes.issuperset(graph.predecessors(child)):
                        current_nodes.append(child)

            if node.operation in {"Import", "Constant"}:
                graph.remove_node(node)
            elif node.operation in {"Assign", "Keyword", "List", "Tuple"}:
                for parent in parents:
                    for child in children:
                        graph.add_edge(parent, child)
                graph.remove_node(node)
            elif node.operation in {"Call", "Subscript"}:
                pass
            else:
                print("Unknown WIR Node Type: {}".format(node))
                assert False

        # By modifying edges, most labels are lost, so we remove the rest of them too
        for (parent, child, edge_attributes) in graph.edges(data=True):
            edge_attributes.clear()

        return graph

    @staticmethod
    def remove_all_non_operators_and_update_names(graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Removes all nodes that can not be a operator we might care about
        """
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
        processed_nodes = set()
        while len(current_nodes) != 0:
            node = current_nodes.pop(0)
            processed_nodes.add(node)
            parents = list(graph.predecessors(node))
            children = list(graph.successors(node))

            # Nodes can have multiple parents, only want to process them once we processed all parents
            for child in children:
                if child not in processed_nodes:
                    if processed_nodes.issuperset(graph.predecessors(child)):
                        current_nodes.append(child)

            if node.module in WirToDagTransformer.OPERATOR_MAP:
                new_dag_vertex = DagVertex(node.node_id, WirToDagTransformer.OPERATOR_MAP[node.module], node.lineno,
                                           node.col_offset, node.module)
                for parent in parents:
                    graph.add_edge(parent, new_dag_vertex)
                for child in children:
                    graph.add_edge(new_dag_vertex, child)
                graph.remove_node(node)
                processed_nodes.add(new_dag_vertex)
            else:
                for parent in parents:
                    for child in children:
                        graph.add_edge(parent, child)
                graph.remove_node(node)

        return graph

    def get_parent_operator_identifier_for_operator(self, lineno, col_offset):
        """
        If we store the annotations of analyzers in a map until the next operator needs it, we need some
        way to identify them and know when we can delete them. When e.g., there is a raw_data.dropna, it needs
        to know that the previous operator was the pd.read_csv. Then it can load the annotations from a map.
        Afterwards, the annotations need to be deleted from the map to avoid unnecessary overhead.
        While we might store the annotations directly in the data frame in the case of pandas, in the case of
        sklearn that is probably not easily possible.
        """
        raise NotImplementedError
