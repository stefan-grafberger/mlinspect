"""
Extract a WIR (Workflow Intermediate Representation) from the AST
"""
import ast
from astmonkey import transformers


class Vertex:
    """
    A WIR Vertex
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

    NOT_FOUND_WIR = Vertex(-1, "FIXME", [], "FIXME")

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
        # pylint: disable=too-many-branches
        for child_ast in ast_node.children:
            self.process_node(child_ast)
        if isinstance(ast_node, ast.Expr):
            pass  # probably nothing to do here
        elif isinstance(ast_node, ast.Assign):
            self.extract_wir_assign(ast_node)
        elif isinstance(ast_node, ast.Load):
            pass  # probably nothing to do here
        elif isinstance(ast_node, ast.Name):
            self.extract_wir_name(ast_node)
        elif isinstance(ast_node, ast.Call):
            self.extract_wir_call(ast_node)
        elif isinstance(ast_node, ast.Attribute):
            pass  # maybe already done, need to check edge cases like member variables
        elif isinstance(ast_node, ast.keyword):
            self.extract_wir_keyword(ast_node)
        elif isinstance(ast_node, ast.Name):
            pass
        elif isinstance(ast_node, ast.Constant):
            self.extract_wir_constant(ast_node)
        elif isinstance(ast_node, ast.Import):
            self.extract_wir_import(ast_node)
        elif isinstance(ast_node, ast.ImportFrom):
            self.extract_wir_import_from(ast_node)
        else:
            print("node")
            print(ast_node)
            # assert False

    def extract_wir_import_from(self, ast_node):
        """
        Creates an import vertex. Stores each imported entity in the dict.
        """
        module_name = ast_node.module
        new_wir_node = Vertex(self.get_next_wir_id(), module_name, [], "Import")
        self.graph.append(new_wir_node)
        for imported_entity_ast in ast_node.children:
            assert isinstance(imported_entity_ast, ast.alias)
            entity_name = imported_entity_ast.name
            self.store_variable_wir_mapping(entity_name, new_wir_node)

    def extract_wir_import(self, ast_node):
        """
        Creates an import vertex. The actual module name and not the alias is used as vertex name.
        """
        alias_ast = ast_node.children[0]
        assert isinstance(alias_ast, ast.alias)
        module_name = alias_ast.name
        if alias_ast.asname:
            alias_name = alias_ast.asname
        else:
            alias_name = module_name
        new_wir_node = Vertex(self.get_next_wir_id(), module_name, [], "Import")
        self.graph.append(new_wir_node)
        self.store_variable_wir_mapping(alias_name, new_wir_node)

    def extract_wir_keyword(self, ast_node):
        """
        Creates a keyword vertex. Keywords are named arguments in function calls.
        """
        child_ast = ast_node.children[0]
        child_wir = self.get_wir_node_for_ast(child_ast)
        new_wir_node = Vertex(self.get_next_wir_id(), ast_node.arg, [child_wir], "Keyword")
        self.graph.append(new_wir_node)
        self.store_ast_node_wir_mapping(ast_node, new_wir_node)

    def extract_wir_assign(self, ast_node):
        """
        Creates an assign vertex and saves the target variable mapping in the dict
        """
        assign_left_ast = ast_node.children[0]
        assign_right_ast = ast_node.children[1]
        assign_right_wir = self.get_wir_node_for_ast(assign_right_ast)
        var_name = assign_left_ast.id
        new_wir_node = Vertex(self.get_next_wir_id(), var_name, [assign_right_wir], "Assign")
        self.graph.append(new_wir_node)
        self.store_variable_wir_mapping(var_name, new_wir_node)

    def extract_wir_constant(self, ast_node):
        """
        Creates a vertex for a constant in the code like a String or number
        """
        new_wir_node = Vertex(self.get_next_wir_id(), str(ast_node.n), [], "Constant_" + str(ast_node.kind))
        self.graph.append(new_wir_node)
        self.store_ast_node_wir_mapping(ast_node, new_wir_node)

    def extract_wir_call(self, ast_node):
        """
        Creates a vertex for a function call in the code
        """
        parent_in_wir_ast_nodes = ast_node.children[1:]
        wir_parents = [self.get_wir_node_for_ast(ast_child) for ast_child in parent_in_wir_ast_nodes]

        name_or_attribute_ast = ast_node.children[0]
        if isinstance(name_or_attribute_ast, ast.Name):
            name = name_or_attribute_ast.id
            possible_import_wir_node = self.get_wir_node_for_variable(name)
            if possible_import_wir_node != WirExtractor.NOT_FOUND_WIR:
                wir_parents.insert(0, possible_import_wir_node)
        elif isinstance(name_or_attribute_ast, ast.Attribute):
            name = name_or_attribute_ast.attr
            object_with_that_func_ast = name_or_attribute_ast.children[0]
            object_with_that_func_wir = self.get_wir_node_for_ast(object_with_that_func_ast)
            wir_parents.insert(0, object_with_that_func_wir)
        else:
            assert False

        new_wir_node = Vertex(self.get_next_wir_id(), name, wir_parents, "Call")
        self.graph.append(new_wir_node)
        self.store_ast_node_wir_mapping(ast_node, new_wir_node)

    def extract_wir_name(self, ast_node):
        """
        Does not create a vertex. A name node can be for loading or storing variables.
        When storing, an Assign is used. The assign-extraction takes care of storing
        variables/wir mapping in the dict, so we do not need to do it here.
        For Loads, it does not create a new vertex but takes care of referencing the
        correct wir vertex when e.g., a call uses this name ast node.
        """
        child_ast = ast_node.children[0]
        if isinstance(child_ast, ast.Store):
            pass  # Assign takes care of saving var then
        elif isinstance(child_ast, ast.Load):
            name = ast_node.id
            wir_node_last_modification = self.get_wir_node_for_variable(name)
            self.store_ast_node_wir_mapping(ast_node, wir_node_last_modification)

    def store_ast_node_wir_mapping(self, ast_entity_name, wir_node) -> None:
        """
        Store which named_object belongs too which wir_node
        """
        self.ast_wir_map[ast_entity_name] = wir_node

    def get_wir_node_for_ast(self, ast_entity_name):
        """
        Get the current wir_node for a named_object
        """
        return self.ast_wir_map.get(ast_entity_name, WirExtractor.NOT_FOUND_WIR)

    def store_variable_wir_mapping(self, variable_name, wir_node) -> None:
        """
        Store which named_object belongs too which wir_node
        """
        self.variable_wir_map[variable_name] = wir_node

    def get_wir_node_for_variable(self, variable_name):
        """
        Get the current wir_node for a named_object
        """
        return self.variable_wir_map.get(variable_name, WirExtractor.NOT_FOUND_WIR)

    def get_next_wir_id(self):
        """
        Get the next id for a wir_node
        """
        wir_id = self.next_node_id
        self.next_node_id += 1
        return wir_id
