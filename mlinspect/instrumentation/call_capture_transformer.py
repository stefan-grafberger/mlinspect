"""
Inserts function call capturing into the DAG
"""
import ast
import astunparse


class CallCaptureTransformer(ast.NodeTransformer):
    """
    ast.NodeTransformer to replace calls with captured calls
    """

    def visit_Call(self, node):
        """
        Instrument all function calls
        """
        # pylint: disable=no-self-use, invalid-name
        ast.NodeTransformer.generic_visit(self, node)

        if hasattr(node.func, "id") and node.func.id == "instrumented_call_used":
            return node

        code = astunparse.unparse(node)

        args = ast.List(node.args, ctx=ast.Load())
        args_code = ast.List([ast.Constant(n=astunparse.unparse(arg).split("\n", 1)[0], kind=None)
                              for arg in node.args], ctx=ast.Load())

        instrumented_call_node = ast.Call(func=ast.Name(id='instrumented_call_used', ctx=ast.Load()),
                                          args=[args, args_code, node, ast.Constant(n=code, kind=None),
                                                ast.Constant(n=node.lineno, kind=None),
                                                ast.Constant(n=node.col_offset, kind=None)],
                                          keywords=[])

        # TODO: warn if unrecognized function call, expressions
        return ast.copy_location(instrumented_call_node, node)
