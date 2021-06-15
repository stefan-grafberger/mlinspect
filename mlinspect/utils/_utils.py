"""
Some useful utils for the project
"""
import ast
from pathlib import Path

import networkx
import numpy
from sklearn.exceptions import NotFittedError
from gensim.sklearn_api import W2VTransformer
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent.parent


# Apparently python AST nodes have no equals, so we need some workarounds to identify ast nodes
def simplify_ast_call_nodes(node: ast):
    """
    Apparently python AST nodes have no equals.
    """
    if not isinstance(node, ast.Call):
        assert False

    id_tuple = (node.lineno, node.col_offset)
    return id_tuple


def get_sorted_node_parents(graph, node_with_parents):
    """
    Get the parent nodes of a WIR node sorted by argument index.
    """
    node_parents = list(graph.predecessors(node_with_parents))
    node_parents_with_arg_index = [(node_parent, graph.get_edge_data(node_parent, node_with_parents))
                                   for node_parent in node_parents]
    sorted_node_parents_with_arg_index = sorted(node_parents_with_arg_index, key=lambda x: x[1]['arg_index'])
    sorted_node_parents = [node_parent[0] for node_parent in sorted_node_parents_with_arg_index]
    return sorted_node_parents


def traverse_graph_and_process_nodes(graph: networkx.DiGraph, func, start_nodes=None, end_node=None, child_filter=None):
    """
    Traverse the WIR node by node from top to bottom
    """
    if not start_nodes:
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
    else:
        current_nodes = start_nodes
    processed_nodes = set()
    while len(current_nodes) != 0:
        node = current_nodes.pop(0)
        processed_nodes.add(node)
        children = list(graph.successors(node))
        if child_filter:
            children = [child for child in children if child_filter(child)]

        # Nodes can have multiple parents, only want to process them once we processed all parents
        if not end_node or node != end_node:
            for child in children:
                if child not in processed_nodes:
                    predecessors = graph.predecessors(child)
                    if child_filter or processed_nodes.issuperset(predecessors):
                        current_nodes.append(child)

        func(node, processed_nodes)
    return graph


class MyW2VTransformer(W2VTransformer):
    """Some custom w2v transformer."""

    def partial_fit(self, X):
        # pylint: disable=useless-super-delegation
        super().partial_fit([X])

    def fit(self, X, y=None):
        X = X.iloc[:, 0].tolist()
        return super().fit([X], y)

    def transform(self, words):
        words = words.iloc[:, 0].tolist()
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        vectors = []
        for word in words:
            if word in self.gensim_model.wv:
                vectors.append(self.gensim_model.wv[word])
            else:
                vectors.append(numpy.zeros(self.size))
        return numpy.reshape(numpy.array(vectors), (len(words), self.size))


class MyKerasClassifier(KerasClassifier):
    """A Keras Wrapper that sets input_dim on fit"""

    def fit(self, x, y, **kwargs):
        """Create and fit a simple neural network"""
        self.sk_params['input_dim'] = x.shape[1]
        super().fit(x, y, **kwargs)
