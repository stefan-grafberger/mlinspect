"""
Extract a WIR (Workflow Intermediate Representation) from the AST
"""
import ast
from astmonkey import transformers


class Vertex:
    """
    A WIR Verrtex
    """
    def __init__(self, parent_vertices, name, operation, keyword_arg_name=None):
        self.parent_vertices = parent_vertices
        self.name = name
        self.operation = operation
        self.keyword_arg_name = keyword_arg_name

    def __repr__(self):
        message = "(vertex_name={}: parent={}, op={}, keyword_arg_name={})" \
            .format(self.name, self.parent_vertices, self.operation, self.keyword_arg_name)
        return message

    def display(self):
        """
        Print the vertex
        """
        message = "(vertex_name={}: parent={}, op={}, keyword_arg_name={})"\
            .format(self.name, self.parent_vertices, self.operation, self.keyword_arg_name)
        print(message)


class WirExtractor:
    """
    Extract WIR (Workflow Intermediate Representation) from the AST
    """

    def __init__(self):
        self.ast_wir_map = {}
        self.variable_wir_map = {}
        self.graph = []

    def extract_wir(self, ast_root: ast.Module):
        """
        Instrument all function calls
        """
        # pylint: disable=no-self-use
        enriched_ast = transformers.ParentChildNodeTransformer().visit(ast_root)

        assert isinstance(enriched_ast, ast.Module)
        self.process_node(enriched_ast)

        return "test"

    def process_node(self, ast_node):
        """
        Recursively generates the WIR
        """
        graph = []
        for child_ast_node in ast_node.children:
            self.process_node(child_ast_node)
        if isinstance(ast_node, ast.Expr): # maybe we can ignore them?
            pass
            # process children, insert children nodes in graph, insert current nodes with children as parent nodes
        elif isinstance(ast_node, ast.Assign):
            pass
            # process children[1], insert in graph, insert children[0] as current node with children as parent nodes
        elif isinstance(ast_node, ast.Call):
            pass
            # process children, insert children nodes in graph, insert current nodes with children as parent nodes
        elif isinstance(ast_node, ast.Attribute):
            pass
            # can be from library or object. source is defined in value. process value and then use it as parent nodes
        elif isinstance(ast_node, ast.Str):
            new_wir_node = Vertex([], ast.Str.s, "String")
            graph.append(new_wir_node)
            self.store_ast_node_wir_mapping(ast_node, new_wir_node)
        elif isinstance(ast_node, ast.keyword):
            new_wir_node = Vertex([], ast_node.arg, "Keyword")
            graph.append(new_wir_node)
            self.store_ast_node_wir_mapping(ast_node, new_wir_node)
        elif isinstance(ast_node, ast.Num):
            new_wir_node = Vertex([], ast_node.s, "String")
            graph.append(new_wir_node)
            self.store_ast_node_wir_mapping(ast_node, new_wir_node)
        elif isinstance(ast_node, ast.Name):
            new_wir_node = Vertex([], ast_node.id, "Name")
            graph.append(new_wir_node)
            self.store_ast_node_wir_mapping(ast_node, new_wir_node)
        elif isinstance(ast_node, ast.NameConstant):
            new_wir_node = Vertex([], ast.Str.s, "NameConstant")
            graph.append(new_wir_node)
            self.store_ast_node_wir_mapping(ast_node, new_wir_node)
        else:
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
        return self.ast_wir_map[ast_entity_name]

    def store_variable_wir_mapping(self, variable_name, wir_node) -> None:
        """
        Store which named_object belongs too which wir_node
        """
        self.variable_wir_map[variable_name] = wir_node

    def get_wir_node_for_variable(self, variable_name):
        """
        Get the current wir_node for a named_object
        """
        return self.variable_wir_map[variable_name]
