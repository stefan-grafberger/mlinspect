"""
Extract a WIR (Workflow Intermediate Representation) from the AST
"""
import ast
from astmonkey import transformers
from mlinspect.utils import simplify_ast_call_nodes


class Vertex:
    """
    A WIR Vertex
    """
    def __init__(self, node_id, name, caller_parent, other_parent_vertices, operation):
        # pylint: disable=too-many-arguments
        self.node_id = node_id
        self.name = name
        self.caller_parent = caller_parent
        self.other_parent_vertices = other_parent_vertices
        self.operation = operation

    def __repr__(self):
        caller_id = self.caller_parent.node_id if self.caller_parent else None
        parent_ids = [vertex.node_id for vertex in self.other_parent_vertices if self.other_parent_vertices]
        message = "(node_id={}: vertex_name={}, caller= {}, parents={}, op={})" \
            .format(self.node_id, self.name, caller_id, parent_ids, self.operation)
        return message

    def display(self):
        """
        Print the vertex
        """
        print(self.__repr__)

    def __eq__(self, other):
        return self.node_id == other.node_id and \
               self.name == other.name and \
               self.caller_parent == other.caller_parent and \
               self.other_parent_vertices == other.other_parent_vertices and \
               self.operation == other.operation


class WirExtractor:
    """
    Extract WIR (Workflow Intermediate Representation) from the AST
    """

    NOT_FOUND_WIR = Vertex(-1, "FIXME", None, [], "FIXME")

    def __init__(self, ast_root: ast.Module):
        self.ast_wir_map = {}
        self.variable_wir_map = {}
        self.graph = []
        self.next_node_id = 0

        self.ast_root = ast_root
        self.ast_call_to_module = None

    def extract_wir(self, ast_call_to_module=None):
        """
        Instrument all function calls
        """
        # pylint: disable=no-self-use
        self.ast_call_to_module = ast_call_to_module

        enriched_ast = transformers.ParentChildNodeTransformer().visit(self.ast_root)
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

        if isinstance(ast_node, ast.Assign):
            self.extract_wir_assign(ast_node)
        elif isinstance(ast_node, ast.Name):
            self.extract_wir_name(ast_node)
        elif isinstance(ast_node, ast.Call):
            self.extract_wir_call(ast_node)
        elif isinstance(ast_node, ast.keyword):
            self.extract_wir_keyword(ast_node)
        elif isinstance(ast_node, ast.Constant):
            self.extract_wir_constant(ast_node)
        elif isinstance(ast_node, ast.Import):
            self.extract_wir_import(ast_node)
        elif isinstance(ast_node, ast.ImportFrom):
            self.extract_wir_import_from(ast_node)
        elif isinstance(ast_node, ast.List):
            self.extract_wir_list(ast_node)
        elif isinstance(ast_node, ast.Subscript):
            self.extract_wir_subscript(ast_node)
        elif isinstance(ast_node, ast.Tuple):
            self.extract_wir_tuple(ast_node)
        elif isinstance(ast_node, (ast.Attribute, ast.Expr, ast.Index, ast.Load, ast.Module, ast.Name, ast.Store,
                                   ast.alias)):
            pass  # TODO: Test if we really covered all necessary edge cases
        else:
            print("AST Node Type not supported yet: {}!".format(str(ast_node)))
            assert False

    def extract_wir_tuple(self, ast_node):
        """
        Creates a tuple vertex.
        """
        parent_in_wir_ast_nodes = ast_node.children[:-1]
        wir_parents = [self.get_wir_node_for_ast(ast_child) for ast_child in parent_in_wir_ast_nodes]
        new_wir_node = Vertex(self.get_next_wir_id(), "as_tuple", None, wir_parents, "Tuple")
        self.graph.append(new_wir_node)
        self.store_ast_node_wir_mapping(ast_node, new_wir_node)

    def extract_wir_subscript(self, ast_node):
        """
        Creates a subscript vertex. Currently only supports index subscripts.
        """
        name_ast = ast_node.children[0]
        assert isinstance(name_ast, ast.Name)  # TODO: Cover other edge cases
        name_name_ast = name_ast.id
        name_wir = self.get_wir_node_for_variable(name_name_ast)
        index_ast = ast_node.children[1]
        assert isinstance(index_ast, ast.Index)
        index_constant_ast = index_ast.children[0]
        assert isinstance(index_constant_ast, ast.Constant)
        constant_wir = self.get_wir_node_for_ast(index_constant_ast)
        new_wir_node = Vertex(self.get_next_wir_id(), "Index-Subscript", name_wir, [constant_wir], "Subscript")
        self.graph.append(new_wir_node)
        self.store_ast_node_wir_mapping(ast_node, new_wir_node)

    def extract_wir_list(self, ast_node):
        """
        Creates a list vertex.
        """
        parent_in_wir_ast_nodes = ast_node.children[:-1]
        wir_parents = [self.get_wir_node_for_ast(ast_child) for ast_child in parent_in_wir_ast_nodes]
        new_wir_node = Vertex(self.get_next_wir_id(), "as_list", None, wir_parents, "List")
        self.graph.append(new_wir_node)
        self.store_ast_node_wir_mapping(ast_node, new_wir_node)

    def extract_wir_import_from(self, ast_node):
        """
        Creates an import vertex. Stores each imported entity in the dict.
        """
        module_name = ast_node.module
        new_wir_node = Vertex(self.get_next_wir_id(), module_name, None, [], "Import")
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
        new_wir_node = Vertex(self.get_next_wir_id(), module_name, None, [], "Import")
        self.graph.append(new_wir_node)
        self.store_variable_wir_mapping(alias_name, new_wir_node)

    def extract_wir_keyword(self, ast_node):
        """
        Creates a keyword vertex. Keywords are named arguments in function calls.
        """
        child_ast = ast_node.children[0]
        child_wir = self.get_wir_node_for_ast(child_ast)
        new_wir_node = Vertex(self.get_next_wir_id(), ast_node.arg, None, [child_wir], "Keyword")
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
        new_wir_node = Vertex(self.get_next_wir_id(), var_name, None, [assign_right_wir], "Assign")
        self.graph.append(new_wir_node)
        self.store_variable_wir_mapping(var_name, new_wir_node)

    def extract_wir_constant(self, ast_node):
        """
        Creates a vertex for a constant in the code like a String or number
        """
        new_wir_node = Vertex(self.get_next_wir_id(), str(ast_node.n), None, [], "Constant")
        self.graph.append(new_wir_node)
        self.store_ast_node_wir_mapping(ast_node, new_wir_node)

    def extract_wir_call(self, ast_node):
        """
        Creates a vertex for a function call in the code
        """
        parent_in_wir_ast_nodes = ast_node.children[1:]
        caller_parent = None
        wir_parents = [self.get_wir_node_for_ast(ast_child) for ast_child in parent_in_wir_ast_nodes]

        name_or_attribute_ast = ast_node.children[0]
        if isinstance(name_or_attribute_ast, ast.Name):
            name = name_or_attribute_ast.id
            possible_import_wir_node = self.get_wir_node_for_variable(name)
            if possible_import_wir_node != WirExtractor.NOT_FOUND_WIR:
                caller_parent = possible_import_wir_node
        elif isinstance(name_or_attribute_ast, ast.Attribute):
            name = name_or_attribute_ast.attr
            object_with_that_func_ast = name_or_attribute_ast.children[0]
            object_with_that_func_wir = self.get_wir_node_for_ast(object_with_that_func_ast)
            caller_parent = object_with_that_func_wir
        else:
            assert False

        if self.ast_call_to_module:  # TODO: Test this and use it
            ast_call_node_lookup_key = simplify_ast_call_nodes(ast_node)
            module = self.ast_call_to_module[ast_call_node_lookup_key]
            print(module)
        new_wir_node = Vertex(self.get_next_wir_id(), name, caller_parent, wir_parents, "Call")
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
