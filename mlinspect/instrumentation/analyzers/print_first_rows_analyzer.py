"""
A simple example analyzer
"""
from numpy import NaN


class PrintFirstRowsAnalyzer:
    """
    A simple example analyzer
    """

    def __init__(self, row_count: int):
        self.row_count = row_count

    def visit_operator(self, operator, row_iterator):
        """
        Visit an operator
        """
        current_count = - 1
        if operator == "Data Source":
            print("Data Source")
            for row in row_iterator:
                current_count += 1
                if current_count < self.row_count:
                    print("{}: {}".format(current_count, row.output))
                if current_count == 0:
                    yield "hello"
                elif current_count == 2:
                    yield "world"
                else:
                    yield current_count
        elif operator == "Selection":
            print("Selection")
            for row in row_iterator:
                current_count += 1
                annotation = row.annotation.get_value_by_column_index(0)
                if annotation in {"hello", "world"} or (annotation <= self.row_count and annotation != NaN):
                    print("{}: {}".format(annotation, row.output))
                yield row.annotation

    def __eq__(self, other):
        return isinstance(other, PrintFirstRowsAnalyzer) and \
               self.row_count == other.row_count
