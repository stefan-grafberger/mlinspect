"""
The Interface for the different instrumentation backends
"""
import abc


class Analyzer(metaclass=abc.ABCMeta):
    """
    The Interface for the different instrumentation backends
    """

    @property
    @abc.abstractmethod
    def analyzer_id(self):
        """The id of the analyzer"""
        raise NotImplementedError

    @abc.abstractmethod
    def visit_operator(self, operator, row_iterator):
        """Visit an operator in the DAG"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_operator_annotation_after_visit(self):
        """Get the output to be included in the DAG"""
        raise NotImplementedError

    def __eq__(self, other):
        """Analyzers must implement equals"""
        return (isinstance(other, self.__class__) and
                self.analyzer_id == other.analyzer_id)

    def __hash__(self):
        """Analyzers must be hashable"""
        return hash((self.__class__.__name__, self.analyzer_id))

    def __repr__(self):
        """Analyzers must be hashable"""
        return "{}({})".format(self.__class__.__name__, self.analyzer_id)
