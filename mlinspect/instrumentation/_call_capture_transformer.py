"""
Inserts function call capturing into the DAG
"""
import ast


class CallCaptureTransformer(ast.NodeTransformer):
    """
    ast.NodeTransformer to replace calls with captured calls
    """

    def __init__(self, source_code):
        super().__init__()
        self.source_code = source_code

    def visit_Call(self, node):
        """
        Instrument all function calls
        """
        # pylint: disable=invalid-name
        ast.NodeTransformer.generic_visit(self, node)
        call_code = ast.get_source_segment(self.source_code, node)

        self.add_before_call_used_value_capturing_call(call_code, node, self.source_code)
        self.add_before_call_used_args_capturing_call(call_code, node, self.source_code)
        self.add_before_call_used_kwargs_capturing_call(call_code, node, self.source_code)
        instrumented_call_node = self.add_after_call_used_capturing(call_code, node, False)

        return instrumented_call_node

    def visit_Subscript(self, node):
        """
        Instrument all subscript calls
        """
        # pylint: disable=invalid-name
        ast.NodeTransformer.generic_visit(self, node)
        if isinstance(node.ctx, ast.Load):
            subscript_code = ast.get_source_segment(self.source_code, node)

            self.add_before_call_used_value_capturing_subscript(subscript_code, node, self.source_code)
            self.add_before_call_used_args_capturing_subscript(subscript_code, node, False, self.source_code)
            instrumented_call_node = self.add_after_call_used_capturing(subscript_code, node, True)

            result = instrumented_call_node
        elif isinstance(node.ctx, ast.Store):
            subscript_code = ast.get_source_segment(self.source_code, node)

            self.add_before_call_used_args_capturing_subscript(subscript_code, node, True, self.source_code)
            result = node
        else:
            assert False
        return result

    @staticmethod
    def add_before_call_used_value_capturing_call(call_code, node, all_source_code):
        """
        When the method of some object is called, capture the value of the object before executing the method
        """
        if hasattr(node.func, "value"):
            old_value_node = node.func.value
            value_code = ast.get_source_segment(all_source_code, old_value_node)
            new_value_node = ast.Call(func=ast.Name(id='before_call_used_value', ctx=ast.Load()),
                                      args=[ast.Constant(n=False, kind=None),
                                            ast.Constant(n=call_code, kind=None),
                                            ast.Constant(n=value_code, kind=None),
                                            old_value_node,
                                            ast.Constant(n=node.lineno, kind=None),
                                            ast.Constant(n=node.col_offset, kind=None),
                                            ast.Constant(n=node.end_lineno, kind=None),
                                            ast.Constant(n=node.end_col_offset, kind=None)],
                                      keywords=[])
            node.func.value = new_value_node

    @staticmethod
    def add_before_call_used_value_capturing_subscript(call_code, node, all_source_code):
        """
        When the __getitem__ method of some object is called, capture the value of the object before executing the
        method
        """
        old_value_node = node.value
        value_code = ast.get_source_segment(all_source_code, old_value_node)
        new_value_node = ast.Call(func=ast.Name(id='before_call_used_value', ctx=ast.Load()),
                                  args=[ast.Constant(n=True, kind=None),
                                        ast.Constant(n=call_code, kind=None),
                                        ast.Constant(n=value_code, kind=None),
                                        old_value_node,
                                        ast.Constant(n=node.lineno, kind=None),
                                        ast.Constant(n=node.col_offset, kind=None),
                                        ast.Constant(n=node.end_lineno, kind=None),
                                        ast.Constant(n=node.end_col_offset, kind=None)],
                                  keywords=[])
        node.value = new_value_node

    @staticmethod
    def add_before_call_used_args_capturing_call(call_code, node, all_source_code):
        """
        When a method is called, capture the arguments of the method before executing it
        """
        old_args_nodes_ast = ast.List(node.args, ctx=ast.Load())
        old_args_code = ast.List([ast.Constant(n=ast.get_source_segment(all_source_code, arg), kind=None)
                                  for arg in node.args], ctx=ast.Load())
        new_args_node = ast.Starred(value=ast.Call(func=ast.Name(id='before_call_used_args', ctx=ast.Load()),
                                                   args=[ast.Constant(n=False, kind=None),
                                                         ast.Constant(n=call_code, kind=None),
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
    def add_before_call_used_args_capturing_subscript(call_code, node, store, all_source_code):
        """
        When the __getitem__ method of some object is called, capture the arguments of the method before executing it
        """
        old_args_code = ast.List([ast.Constant(n=ast.get_source_segment(all_source_code, node.slice.value), kind=None)],
                                 ctx=ast.Load())
        new_args_node = ast.Call(func=ast.Name(id='before_call_used_args', ctx=ast.Load()),
                                 args=[ast.Constant(n=True, kind=None),
                                       ast.Constant(n=call_code, kind=None),
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
    def add_before_call_used_kwargs_capturing_call(call_code, node, all_source_code):
        """
        When a method is called, capture the keyword arguments of the method before executing it
        """
        old_kwargs_nodes_ast = node.keywords  # old_kwargs_nodes_ast = ast.List(node.keywords, ctx=ast.Load())
        old_kwargs_code = ast.List([ast.Constant(n=ast.get_source_segment(all_source_code, kwarg), kind=None)
                                    for kwarg in node.keywords], ctx=ast.Load())
        new_kwargs_node = ast.keyword(value=ast.Call(func=ast.Name(id='before_call_used_kwargs', ctx=ast.Load()),
                                                     args=[ast.Constant(n=False, kind=None),
                                                           ast.Constant(n=call_code, kind=None),
                                                           old_kwargs_code,
                                                           ast.Constant(n=node.lineno, kind=None),
                                                           ast.Constant(n=node.col_offset, kind=None),
                                                           ast.Constant(n=node.end_lineno, kind=None),
                                                           ast.Constant(n=node.end_col_offset, kind=None)],
                                                     keywords=old_kwargs_nodes_ast), arg=None)
        node.keywords = [new_kwargs_node]

    @staticmethod
    def add_after_call_used_capturing(subscript_code, node, subscript):
        """
        After a method got executed, capture the return value
        """
        instrumented_call_node = ast.Call(func=ast.Name(id='after_call_used', ctx=ast.Load()),
                                          args=[ast.Constant(n=subscript, kind=None),
                                                ast.Constant(n=subscript_code, kind=None),
                                                node,
                                                ast.Constant(n=node.lineno, kind=None),
                                                ast.Constant(n=node.col_offset, kind=None),
                                                ast.Constant(n=node.end_lineno, kind=None),
                                                ast.Constant(n=node.end_col_offset, kind=None)],
                                          keywords=[])
        instrumented_call_node = ast.copy_location(instrumented_call_node, node)
        return instrumented_call_node
