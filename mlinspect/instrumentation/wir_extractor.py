"""
Extract a WIR (Workflow Intermediate Representation) from the AST
"""
import ast
from astmonkey import transformers


class Vertex():
    """
    A WIR Verrtex
    """
    def __init__(self, node_id, name, parent_vertices, operation):
        self.node_id = node_id
        self.name = name
        self.parent_vertices = parent_vertices
        self.operation = operation

    def __repr__(self):
        parent_ids = [vertex.node_id for vertex in self.parent_vertices if self.parent_vertices]
        message = "(node_id={}: vertex_name={}, parent={}, op={})" \
            .format(self.node_id, self.name, parent_ids, self.operation)
        return message

    def display(self):
        """
        Print the vertex
        """
        parent_ids = [vertex.node_id for vertex in self.parent_vertices if self.parent_vertices]
        message = "(node_id={}: vertex_name={}, parent={}, op={})" \
            .format(self.node_id, self.name, parent_ids, self.operation)
        print(message)

    def __eq__(self, other):
        return self.node_id == other.node_id and \
               self.name == other.name and \
               self.parent_vertices == other.parent_vertices and \
               self.operation == other.operation


class WirExtractor:
    """
    Extract WIR (Workflow Intermediate Representation) from the AST
    """

    def __init__(self):
        self.ast_wir_map = {}
        self.variable_wir_map = {}
        self.graph = []
        self.next_node_id = 0

    def extract_wir(self, ast_root: ast.Module):
        """
        Instrument all function calls
        """
        # pylint: disable=no-self-use
        enriched_ast = transformers.ParentChildNodeTransformer().visit(ast_root)

        assert isinstance(enriched_ast, ast.Module)
        self.process_node(enriched_ast)

        return self.graph

    def process_node(self, ast_node):
        """
        Recursively generates the WIR
        """
        for child_ast_node in ast_node.children:
            self.process_node(child_ast_node)
        if isinstance(ast_node, ast.Expr): # maybe we can ignore them?
            pass
            # process children, insert children nodes in graph, insert current nodes with children as parent nodes
        elif isinstance(ast_node, ast.Assign):
            pass
            # process children[1], insert in graph, insert children[0] as current node with children as parent nodes
        elif isinstance(ast_node, ast.Call):
            name_or_attribute_ast = ast_node.children[0]
            if isinstance(name_or_attribute_ast, ast.Name):
                name = name_or_attribute_ast.id
            elif isinstance(name_or_attribute_ast, ast.Attribute):
                name = name_or_attribute_ast.attr  # FIXME: need to construct WIR parents correctly
            else:
                assert False
            parent_in_wir_ast_nodes = ast_node.children[1:]

            wir_parents = [self.get_wir_node_for_ast(ast_child) for ast_child in parent_in_wir_ast_nodes]
            new_wir_node = Vertex(self.get_next_wir_id(), name, wir_parents, "Call")
            self.graph.append(new_wir_node)
            self.store_ast_node_wir_mapping(ast_node, new_wir_node)

            # process children, insert children nodes in graph, insert current nodes with children as parent nodes
        elif isinstance(ast_node, ast.Attribute):
            pass
            # can be from library or object. source is defined in value. process value and then use it as parent nodes
        elif isinstance(ast_node, ast.keyword):
            new_wir_node = Vertex(self.get_next_wir_id(), ast_node.arg, [], "Keyword")
            self.graph.append(new_wir_node)
            self.store_ast_node_wir_mapping(ast_node, new_wir_node)
        elif isinstance(ast_node, ast.Name):
            pass
        elif isinstance(ast_node, ast.Constant):
            new_wir_node = Vertex(self.get_next_wir_id(), str(ast_node.n), [], "Constant_" + str(ast_node.kind))
            self.graph.append(new_wir_node)
            self.store_ast_node_wir_mapping(ast_node, new_wir_node)
        else:
            print("node")
            print(ast_node)
            # assert False

    def store_ast_node_wir_mapping(self, ast_entity_name, wir_node) -> None:
        """
        Store which named_object belongs too which wir_node
        """
        self.ast_wir_map[ast_entity_name] = wir_node

    def get_wir_node_for_ast(self, ast_entity_name):
        """
        Get the current wir_node for a named_object
        """
        return self.ast_wir_map.get(ast_entity_name, Vertex(-1, "FIXME", [], "FIXME"))

    def store_variable_wir_mapping(self, variable_name, wir_node) -> None:
        """
        Store which named_object belongs too which wir_node
        """
        self.variable_wir_map[variable_name] = wir_node

    def get_wir_node_for_variable(self, variable_name):
        """
        Get the current wir_node for a named_object
        """
        return self.variable_wir_map.get(variable_name, Vertex(-1, "FIXME", [], "FIXME"))

    def get_next_wir_id(self):
        """
        Get the next id for a wir_node
        """
        wir_id = self.next_node_id
        self.next_node_id += 1
        return wir_id
