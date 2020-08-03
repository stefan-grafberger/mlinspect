"""
The Interface for the different instrumentation backends
"""
import abc


class Backend(metaclass=abc.ABCMeta):
    """
    The Interface for the different instrumentation backends
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'prefix') and
                hasattr(subclass, 'call_description_map') and
                callable(subclass.call_description_map) and
                hasattr(subclass, 'before_call_used_value') and
                callable(subclass.before_call_used_value) and
                hasattr(subclass, 'before_call_used_args') and
                callable(subclass.before_call_used_args) and
                hasattr(subclass, 'before_call_used_kwargs') and
                callable(subclass.before_call_used_kwargs) and
                hasattr(subclass, 'after_call_used') and
                callable(subclass.after_call_used))

    @property
    def call_description_map(self):
        """The map with additional runtime descriptions for operators"""
        raise NotImplementedError

    @abc.abstractmethod
    def before_call_used_value(self, function_info, subscript, call_code, value_code, value_value, ast_lineno,
                               ast_col_offset):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError

    @abc.abstractmethod
    def before_call_used_args(self, function_info, subscript, call_code, args_code, ast_lineno, ast_col_offset,
                              args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError

    @abc.abstractmethod
    def before_call_used_kwargs(self, function_info, subscript, call_code, kwargs_code, ast_lineno, ast_col_offset,
                                kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError

    @abc.abstractmethod
    def after_call_used(self, function_info, subscript, call_code, return_value, ast_lineno,
                        ast_col_offset):
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError
