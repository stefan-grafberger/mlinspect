"""
The Vertices used in the WIR as nodes for the networkx.DiGraph
"""


class WirVertex:
    """
    A WIR Vertex
    """

    def __init__(self, node_id, name, operation, lineno=None, col_offset=None, module=None, description=None):
        # pylint: disable=too-many-arguments
        self.node_id = node_id
        self.name = name
        self.operation = operation
        self.lineno = lineno
        self.col_offset = col_offset
        self.module = module
        self.description = description

    def __repr__(self):
        message = "WirVertex(node_id={}: name='{}', operation='{}', lineno={}, col_offset={}, module={}, " \
                  "description='{}')".format(self.node_id, self.name, self.operation, self.lineno,
                                             self.col_offset, self.module, self.description)
        return message

    def __eq__(self, other):
        return isinstance(other, WirVertex) and \
               self.node_id == other.node_id and \
               self.name == other.name and \
               self.operation == other.operation and \
               self.lineno == other.lineno and \
               self.col_offset == other.col_offset and \
               self.module == other.module and \
               self.description == other.description

    def __hash__(self):
        return hash(self.node_id)
