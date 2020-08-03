"""
A simple example analyzer
"""


class PrintFirstRowsAnalyzer:
    """
    A WIR Vertex
    """

    def __init__(self, row_count: int):
        self.row_count = row_count

    def visit_operator(self, row_iterator):
        current_count = 0
        for row in row_iterator:
            if current_count < self.row_count:
                print(row)
            current_count += 1
            yield current_count

    def __eq__(self, other):
        return isinstance(other, PrintFirstRowsAnalyzer) and \
               self.row_count == other.row_count
