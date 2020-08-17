"""
A wrapper for numpy ndarrays to store our additional annotations
"""
import numpy


class MlinspectNdarray(numpy.ndarray):
    """
    A wrapper for numpy ndarrays to store our additional annotations.
    See https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
    """

    def __new__(cls, input_array, annotations=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.annotations = annotations
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # pylint: disable=attribute-defined-outside-init
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.info = getattr(obj, 'annotations', None)
