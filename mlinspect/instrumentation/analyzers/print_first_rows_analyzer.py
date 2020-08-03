"""
A simple example analyzer
"""


class PrintFirstRowsAnalyzer:
    """
    A WIR Vertex
    """

    def __init__(self, row_count: int):
        self.row_count = row_count

    def visit_operator(self, iterator):
        current_count = 0
        for item in iterator:
            if current_count < self.row_count:
                print(item)
            current_count += 1
        return current_count

    def __eq__(self, other):
        return isinstance(other, PrintFirstRowsAnalyzer) and \
               self.row_count == other.row_count
