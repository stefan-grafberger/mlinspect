"""
The Interface for the different instrumentation backends
"""
import abc


class Backend(metaclass=abc.ABCMeta):
    """
    The Interface for the different instrumentation backends
    """

    def __init__(self):
        self.call_description_map = {}
        self.call_analyzer_output_map = {}
        self.analyzers = []

    @property
    @abc.abstractmethod
    def prefix(self):
        """The prefix of the module of the library the backend is for"""
        raise NotImplementedError

    @abc.abstractmethod
    def before_call_used_value(self, function_info, subscript, call_code, value_code, value_value,
                               ast_lineno, ast_col_offset):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError

    @abc.abstractmethod
    def before_call_used_args(self, function_info, subscript, call_code, args_code, ast_lineno,
                              ast_col_offset, args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError

    @abc.abstractmethod
    def before_call_used_kwargs(self, function_info, subscript, call_code, kwargs_code, ast_lineno,
                                ast_col_offset, kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError

    @abc.abstractmethod
    def after_call_used(self, function_info, subscript, call_code, return_value, ast_lineno,
                        ast_col_offset):
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument
        raise NotImplementedError
