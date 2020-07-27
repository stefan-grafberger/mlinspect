"""
Some util functions used in other tests
"""
import networkx

from mlinspect.instrumentation.dag_vertex import DagVertex


def get_expected_dag_adult_easy_py():
    """
    Get the expected DAG for the adult_easy pipeline
    """
    expected_graph = networkx.DiGraph()

    expected_data_source = DagVertex(18, "Data Source", 12, 11, ('pandas.io.parsers', 'read_csv'))
    expected_graph.add_node(expected_data_source)

    expected_select = DagVertex(20, "Selection", 14, 7, ('pandas.core.frame', 'dropna'))
    expected_graph.add_edge(expected_data_source, expected_select)

    expected_project = DagVertex(23, "Projection", 16, 38, ('pandas.core.frame', '__getitem__'))
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagVertex(28, "Projection (Modify)", 16, 9,
                                        ('sklearn.preprocessing._label', 'label_binarize'))
    expected_graph.add_edge(expected_project, expected_project_modify)

    return expected_graph


def get_expected_dag_adult_easy_ipynb():
    """
    Get the expected DAG for the adult_easy pipeline
    """
    expected_graph = networkx.DiGraph()

    expected_data_source = DagVertex(18, "Data Source", 18, 11, ('pandas.io.parsers', 'read_csv'))
    expected_graph.add_node(expected_data_source)

    expected_select = DagVertex(20, "Selection", 20, 7, ('pandas.core.frame', 'dropna'))
    expected_graph.add_edge(expected_data_source, expected_select)

    expected_project = DagVertex(23, "Projection", 22, 38, ('pandas.core.frame', '__getitem__'))
    expected_graph.add_edge(expected_select, expected_project)

    expected_project_modify = DagVertex(28, "Projection (Modify)", 22, 9,
                                        ('sklearn.preprocessing._label', 'label_binarize'))
    expected_graph.add_edge(expected_project, expected_project_modify)

    return expected_graph
