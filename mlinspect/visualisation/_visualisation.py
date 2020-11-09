"""
Utility functions to visualise the extracted DAG
"""
from inspect import cleandoc

import networkx
from networkx.drawing.nx_agraph import to_agraph


def save_fig_to_path(extracted_dag, filename):
    """
    Create a figure of the extracted DAG and save it with some filename
    """

    def get_new_node_label(node):
        label = cleandoc("""
                {} (L{})
                {}
                """.format(node.operator_type.value, node.code_reference.lineno, node.description or ""))
        return label

    # noinspection PyTypeChecker
    extracted_dag = networkx.relabel_nodes(extracted_dag, get_new_node_label)

    agraph = to_agraph(extracted_dag)
    agraph.layout('dot')
    agraph.draw(filename)


def get_dag_as_pretty_string(extracted_dag):
    """
    Create a figure of the extracted DAG and save it with some filename
    """

    def get_new_node_label(node):
        description = ""
        if node.description:
            description = "({})".format(node.description)

        label = "{}{}".format(node.operator_type.value, description)
        return label

    # noinspection PyTypeChecker
    extracted_dag = networkx.relabel_nodes(extracted_dag, get_new_node_label)

    agraph = to_agraph(extracted_dag)
    return agraph.to_string()
