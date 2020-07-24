"""
Inserts function call capturing into the DAG
"""
import ast
import astunparse


class CallCaptureTransformer(ast.NodeTransformer):
    """
    ast.NodeTransformer to replace calls with captured calls
    """
    def __init__(self):
        # Necessary to avoid issues with nested function calls
        # lineno and col_offset are used for identification
        self.already_instrumented_code = set()

    def visit_Call(self, node):
        """
        Instrument all function calls
        """
        # pylint: disable=no-self-use, invalid-name
        # FIXME: See output of function_subscript_index_info_extraction test. os.path.join and str
        # cause get_project_root to be called many times. Maybe I need to use assigns or something to use references
        # or something similar.

        # This is some work in progress testing that does not work. maybe look at the ast? what does it look like
        # after the instrumentation? Everything already works for non-nested calls. In the worst case,
        # we need to un-nest it with assigns. However, that should not be necessary.
        if self.already_instrumented_code.__contains__((node.lineno, node.col_offset)):
            return node

        self.already_instrumented_code.add((node.lineno, node.col_offset))
        ast.NodeTransformer.generic_visit(self, node)
        code = astunparse.unparse(node)

        args = ast.List(node.args, ctx=ast.Load())
        args_code = ast.List([ast.Constant(n=astunparse.unparse(arg).split("\n", 1)[0], kind=None)
                              for arg in node.args], ctx=ast.Load())

        instrumented_call_node = ast.Call(func=ast.Name(id='instrumented_call_used', ctx=ast.Load()),
                                          args=[ast.Constant(n=False, kind=None),
                                                args,
                                                args_code,
                                                node,
                                                ast.Constant(n=code, kind=None),
                                                ast.Constant(n=node.lineno, kind=None),
                                                ast.Constant(n=node.col_offset, kind=None)],
                                          keywords=[])
        # TODO: warn if unrecognized function call, expressions
        return ast.copy_location(instrumented_call_node, node)

    def visit_Subscript(self, node):
        """
        Instrument all subscript calls
        """
        # pylint: disable=no-self-use, invalid-name
        ast.NodeTransformer.generic_visit(self, node)

        code = astunparse.unparse(node)

        args = [node.value, node.slice.value]
        args_ast = ast.List(args, ctx=ast.Load())
        args_code = ast.List([ast.Constant(n=astunparse.unparse(arg).split("\n", 1)[0], kind=None)
                              for arg in args], ctx=ast.Load())

        instrumented_call_node = ast.Call(func=ast.Name(id='instrumented_call_used', ctx=ast.Load()),
                                          args=[ast.Constant(n=True, kind=None),
                                                args_ast,
                                                args_code,
                                                node,
                                                ast.Constant(n=code, kind=None),
                                                ast.Constant(n=node.lineno, kind=None),
                                                ast.Constant(n=node.col_offset, kind=None)],
                                          keywords=[])

        # TODO: warn if unrecognized function call, expressions
        return ast.copy_location(instrumented_call_node, node)
