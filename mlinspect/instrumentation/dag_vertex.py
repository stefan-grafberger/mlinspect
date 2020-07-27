"""
The Vertices used in the WIR as nodes for the networkx.DiGraph
"""


class DagVertex:
    """
    A WIR Vertex
    """

    def __init__(self, node_id, name, lineno=None, col_offset=None, module=None):
        # pylint: disable=too-many-arguments
        self.node_id = node_id
        self.name = name
        self.lineno = lineno
        self.col_offset = col_offset
        self.module = module

    def __repr__(self):
        message = "(node_id={}: vertex_name='{}', lineno={}, col_offset={}, module={})" \
            .format(self.node_id, self.name, self.lineno, self.col_offset, self.module)
        return message

    def display(self):
        """
        Print the vertex
        """
        print(self.__repr__)

    def __eq__(self, other):
        return self.node_id == other.node_id and \
               self.name == other.name and \
               self.lineno == other.lineno and \
               self.col_offset == other.col_offset and \
               self.module == other.module

    def __hash__(self):
        return hash(self.node_id)
