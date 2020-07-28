"""
The Vertices used in the WIR as nodes for the networkx.DiGraph
"""


class DagVertex:
    """
    A WIR Vertex
    """

    def __init__(self, node_id, operator_name, lineno=None, col_offset=None, module=None):
        # pylint: disable=too-many-arguments
        self.node_id = node_id
        self.operator_name = operator_name
        self.module = module
        self.lineno = lineno
        self.col_offset = col_offset

    def __repr__(self):
        message = "DagVertex(node_id={}: operator_name='{}', module={}, lineno={}, col_offset={})" \
            .format(self.node_id, self.operator_name, self.module, self.lineno, self.col_offset)
        return message

    def __eq__(self, other):
        return isinstance(other, DagVertex) and \
               self.node_id == other.node_id and \
               self.operator_name == other.operator_name and \
               self.module == other.module and \
               self.lineno == other.lineno and \
               self.col_offset == other.col_offset

    def __hash__(self):
        return hash(self.node_id)
