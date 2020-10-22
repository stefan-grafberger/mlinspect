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
        # pylint: disable=invalid-name
        ast.NodeTransformer.generic_visit(self, node)
        code = astunparse.unparse(node)

        self.add_before_call_used_value_capturing_call(code, node)
        self.add_before_call_used_args_capturing_call(code, node)
        self.add_before_call_used_kwargs_capturing_call(code, node)
        instrumented_call_node = self.add_after_call_used_capturing(code, node, False)

        return instrumented_call_node

    def visit_Subscript(self, node):
        """
        Instrument all subscript calls
        """
        # pylint: disable=invalid-name
        ast.NodeTransformer.generic_visit(self, node)
        if isinstance(node.ctx, ast.Load):
            code = astunparse.unparse(node)

            self.add_before_call_used_value_capturing_subscript(code, node)
            self.add_before_call_used_args_capturing_subscript(code, node, False)
            instrumented_call_node = self.add_after_call_used_capturing(code, node, True)

            result = instrumented_call_node
        elif isinstance(node.ctx, ast.Store):
            code = astunparse.unparse(node)

            self.add_before_call_used_args_capturing_subscript(code, node, True)
            result = node
        else:
            assert False
        return result


    @staticmethod
    def add_before_call_used_value_capturing_call(code, node):
        """
        When the method of some object is called, capture the value of the object before executing the method
        """
        if hasattr(node.func, "value"):
            old_value_node = node.func.value
            value_code = astunparse.unparse(old_value_node)
            new_value_node = ast.Call(func=ast.Name(id='before_call_used_value', ctx=ast.Load()),
                                      args=[ast.Constant(n=False, kind=None),
                                            ast.Constant(n=code, kind=None),
                                            ast.Constant(n=value_code, kind=None),
                                            old_value_node,
                                            ast.Constant(n=node.lineno, kind=None),
                                            ast.Constant(n=node.col_offset, kind=None),
                                            ast.Constant(n=node.end_lineno, kind=None),
                                            ast.Constant(n=node.end_col_offset, kind=None)],
                                      keywords=[])
            node.func.value = new_value_node

    @staticmethod
    def add_before_call_used_value_capturing_subscript(code, node):
        """
        When the __getitem__ method of some object is called, capture the value of the object before executing the
        method
        """
        old_value_node = node.value
        value_code = astunparse.unparse(old_value_node)
        new_value_node = ast.Call(func=ast.Name(id='before_call_used_value', ctx=ast.Load()),
                                  args=[ast.Constant(n=True, kind=None),
                                        ast.Constant(n=code, kind=None),
                                        ast.Constant(n=value_code, kind=None),
                                        old_value_node,
                                        ast.Constant(n=node.lineno, kind=None),
                                        ast.Constant(n=node.col_offset, kind=None),
                                        ast.Constant(n=node.end_lineno, kind=None),
                                        ast.Constant(n=node.end_col_offset, kind=None)],
                                  keywords=[])
        node.value = new_value_node

    @staticmethod
    def add_before_call_used_args_capturing_call(code, node):
        """
        When a method is called, capture the arguments of the method before executing it
        """
        old_args_nodes_ast = ast.List(node.args, ctx=ast.Load())
        old_args_code = ast.List([ast.Constant(n=astunparse.unparse(arg).split("\n", 1)[0], kind=None)
                                  for arg in node.args], ctx=ast.Load())
        new_args_node = ast.Starred(value=ast.Call(func=ast.Name(id='before_call_used_args', ctx=ast.Load()),
                                                   args=[ast.Constant(n=False, kind=None),
                                                         ast.Constant(n=code, kind=None),
                                                         old_args_code,
                                                         ast.Constant(n=node.lineno, kind=None),
                                                         ast.Constant(n=node.col_offset, kind=None),
                                                         ast.Constant(n=node.end_lineno, kind=None),
                                                         ast.Constant(n=node.end_col_offset, kind=None),
                                                         ast.Constant(n=False, kind=None),
                                                         old_args_nodes_ast],
                                                   keywords=[]), ctx=ast.Load())
        node.args = [new_args_node]

    @staticmethod
    def add_before_call_used_args_capturing_subscript(code, node, store):
        """
        When the __getitem__ method of some object is called, capture the arguments of the method before executing it
        """
        old_args_code = ast.List([ast.Constant(n=astunparse.unparse(node.slice.value).split("\n", 1)[0], kind=None)],
                                 ctx=ast.Load())
        new_args_node = ast.Call(func=ast.Name(id='before_call_used_args', ctx=ast.Load()),
                                 args=[ast.Constant(n=True, kind=None),
                                       ast.Constant(n=code, kind=None),
                                       old_args_code,
                                       ast.Constant(n=node.lineno, kind=None),
                                       ast.Constant(n=node.col_offset, kind=None),
                                       ast.Constant(n=node.end_lineno, kind=None),
                                       ast.Constant(n=node.end_col_offset, kind=None),
                                       ast.Constant(n=store, kind=None),
                                       node.slice.value],
                                 keywords=[])
        node.slice.value = new_args_node

    @staticmethod
    def add_before_call_used_kwargs_capturing_call(code, node):
        """
        When a method is called, capture the keyword arguments of the method before executing it
        """
        old_kwargs_nodes_ast = node.keywords  # old_kwargs_nodes_ast = ast.List(node.keywords, ctx=ast.Load())
        old_kwargs_code = ast.List([ast.Constant(n=astunparse.unparse(kwarg), kind=None)
                                    for kwarg in node.keywords], ctx=ast.Load())
        new_kwargs_node = ast.keyword(value=ast.Call(func=ast.Name(id='before_call_used_kwargs', ctx=ast.Load()),
                                                     args=[ast.Constant(n=False, kind=None),
                                                           ast.Constant(n=code, kind=None),
                                                           old_kwargs_code,
                                                           ast.Constant(n=node.lineno, kind=None),
                                                           ast.Constant(n=node.col_offset, kind=None),
                                                           ast.Constant(n=node.end_lineno, kind=None),
                                                           ast.Constant(n=node.end_col_offset, kind=None)],
                                                     keywords=old_kwargs_nodes_ast), arg=None)
        node.keywords = [new_kwargs_node]

    @staticmethod
    def add_after_call_used_capturing(code, node, subscript):
        """
        After a method got executed, capture the return value
        """
        instrumented_call_node = ast.Call(func=ast.Name(id='after_call_used', ctx=ast.Load()),
                                          args=[ast.Constant(n=subscript, kind=None),
                                                ast.Constant(n=code, kind=None),
                                                node,
                                                ast.Constant(n=node.lineno, kind=None),
                                                ast.Constant(n=node.col_offset, kind=None),
                                                ast.Constant(n=node.end_lineno, kind=None),
                                                ast.Constant(n=node.end_col_offset, kind=None)],
                                          keywords=[])
        instrumented_call_node = ast.copy_location(instrumented_call_node, node)
        return instrumented_call_node
