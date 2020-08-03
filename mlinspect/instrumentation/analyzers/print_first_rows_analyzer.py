"""
A simple example analyzer
"""


class PrintFirstRowsAnalyzer:
    """
    A WIR Vertex
    """

    def __init__(self, row_count: int):
        self.row_count = row_count
        self.current_count = 0

    def start_operator(self, _):
        self.current_count = 0

    def process_row(self, row):
        if self.current_count < self.row_count:
            print(row)
        self.current_count += 1

    def __eq__(self, other):
        return isinstance(other, PrintFirstRowsAnalyzer) and \
               self.row_count == other.row_count
