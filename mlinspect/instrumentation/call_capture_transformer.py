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

        code = astunparse.unparse(node)
        args = ast.List(node.args, ctx=ast.Load())
        args_code = ast.List([ast.Str(astunparse.unparse(arg).split("\n", 1)[0]) for arg in node.args],
                             ctx=ast.Load())

        instrumented_call_node = ast.Call(func=ast.Name(id='instrumented_call_used', ctx=ast.Load()),
                                          args=[args, args_code, node, ast.Str(s=code)], keywords=[])
        ast.copy_location(instrumented_call_node, node)

        # TODO: warn if unrecognized function call, expressions
        return instrumented_call_node
