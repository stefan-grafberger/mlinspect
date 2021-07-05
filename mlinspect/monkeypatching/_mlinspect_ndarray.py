"""
Monkey patching for numpy
"""
import numpy


class MlinspectNdarray(numpy.ndarray):
    """
    A wrapper for numpy ndarrays to store our additional annotations.
    See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    """

    def __new__(cls, input_array, _mlinspect_dag_node=None, _mlinspect_annotation=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj._mlinspect_dag_node = _mlinspect_dag_node
        obj._mlinspect_annotation = _mlinspect_annotation
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # pylint: disable=attribute-defined-outside-init
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._mlinspect_dag_node = getattr(obj, '_mlinspect_dag_node', None)
        self._mlinspect_annotation = getattr(obj, '_mlinspect_annotation', None)

    def ravel(self, order='C'):
        # pylint: disable=no-member
        result = super().ravel(order)
        assert isinstance(result, MlinspectNdarray)
        result._mlinspect_dag_node = self._mlinspect_dag_node  # pylint: disable=protected-access
        result._mlinspect_annotation = self._mlinspect_annotation  # pylint: disable=protected-access
        return result
