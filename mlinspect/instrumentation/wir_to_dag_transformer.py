"""
Extract a DAG from the WIR (Workflow Intermediate Representation)
"""
import networkx


class WirToDagTransformer:
    """
    Extract DAG from the WIR (Workflow Intermediate Representation)
    """

    @staticmethod
    def remove_all_nodes_but_calls_and_subscripts(graph: networkx.DiGraph) -> networkx.DiGraph:
        """
        Removes all nodes that can not be a operator we might care about
        """
        # Useful link:
        # https://stackoverflow.com/questions/61914713/removing-a-node-from-digraph-in-networkx-while-preserving-child-nodes-and-remapp
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
        while len(current_nodes) != 0:
            for node in current_nodes:
                if node.operation == "Constant":
                    successors = graph.successors(node)
                    for successor in successors:
                        print("contract_node {} and {}".format(node, successor))
                    current_nodes.remove(node)
                else:
                    # assert False
                    current_nodes.remove(node)
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
