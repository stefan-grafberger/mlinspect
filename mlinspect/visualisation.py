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
                {}: {}
                {}
                {}
                """.format(node.node_id, node.operator_name, node.module[0], node.module[1]))
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
        label = "{}: {}({}, {})".format(node.node_id, node.operator_name, node.module[0], node.module[1])
        return label

    # noinspection PyTypeChecker
    extracted_dag = networkx.relabel_nodes(extracted_dag, get_new_node_label)

    agraph = to_agraph(extracted_dag)
    return agraph.to_string()
