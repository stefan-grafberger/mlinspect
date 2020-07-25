"""
The Vertices used in the WIR as nodes for the networkx.DiGraph
"""


class WirVertex:
    """
    A WIR Vertex
    """

    def __init__(self, node_id, name, operation, module=None):
        self.node_id = node_id
        self.name = name
        self.operation = operation
        self.module = module

    def __repr__(self):
        message = "(node_id={}: vertex_name='{}', op='{}', module={})" \
            .format(self.node_id, self.name, self.operation, self.module)
        return message

    def display(self):
        """
        Print the vertex
        """
        print(self.__repr__)

    def __eq__(self, other):
        return self.node_id == other.node_id and \
               self.name == other.name and \
               self.operation == other.operation and \
               self.module == other.module

    def __hash__(self):
        return hash(self.node_id)
