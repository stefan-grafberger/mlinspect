"""
Extract a WIR (Workflow Intermediate Representation) from the AST
"""
import ast
import astunparse
from astmonkey import transformers


class WirExtractor:
    """
    Extract WIR (Workflow Intermediate Representation) from the AST
    """

    def __init__(self):
        self.ast_wir_map = {}

    def extract_wir(self, ast_root: ast.Module):
        """
        Instrument all function calls
        """
        # pylint: disable=no-self-use
        enriched_ast = transformers.ParentChildNodeTransformer().visit(ast_root)

        code = astunparse.unparse(enriched_ast)

        print(code)
        return "test"

    def store_node_mapping(self, entity_name, wir_node) -> None:
        """
        Store which named_object belongs too which wir_node
        """
        self.ast_wir_map[entity_name] = wir_node

    def get_wir_node(self, entity_name):
        """
        Get the current wir_node for a named_object
        """
        return self.ast_wir_map[entity_name]
