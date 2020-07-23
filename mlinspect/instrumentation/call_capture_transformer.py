"""
Inserts function call capturing into the DAG
"""
import ast
import astunparse
from mlinspect.utils import simplify_ast_call_nodes


class CallCaptureTransformer(ast.NodeTransformer):
    """
    ast.NodeTransformer to replace calls with captured calls
    """

    def __init__(self):
        self.id_to_call_ast = {}
        self.next_ast_node_id = 0

    def visit_Call(self, node):
        """
        Instrument all function calls
        """
        # pylint: disable=no-self-use, invalid-name
        node.args = [self.visit(arg_node) for arg_node in node.args]

        code = astunparse.unparse(node)
        args = ast.List(node.args, ctx=ast.Load())
        args_code = ast.List([ast.Constant(n=astunparse.unparse(arg).split("\n", 1)[0], kind=None)
                              for arg in node.args], ctx=ast.Load())

        ast_node_id = self.get_next_ast_node_id()
        self.id_to_call_ast[ast_node_id] = simplify_ast_call_nodes(node)

        instrumented_call_node = ast.Call(func=ast.Name(id='instrumented_call_used', ctx=ast.Load()),
                                          args=[args, args_code, node, ast.Constant(n=code, kind=None),
                                                ast.Constant(n=ast_node_id, kind=None)],
                                          keywords=[])
        # TODO: warn if unrecognized function call, expressions
        return instrumented_call_node

    def get_id_to_call_ast(self):
        return self.id_to_call_ast

    def get_next_ast_node_id(self):
        """
        Get the next id for a wir_node
        """
        ast_node_id = self.next_ast_node_id
        self.next_ast_node_id += 1
        return ast_node_id

