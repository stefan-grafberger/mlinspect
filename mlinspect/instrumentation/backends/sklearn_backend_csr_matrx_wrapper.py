"""
A wrapper for numpy ndarrays to store our additional annotations
"""
from scipy.sparse import csr_matrix


class MlinspectCsrMatrix(csr_matrix):
    """
    A wrapper for numpy ndarrays to store our additional annotations.
    See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    """

    def __init__(self):
        super.__init__(self)
        self.annotations = None
