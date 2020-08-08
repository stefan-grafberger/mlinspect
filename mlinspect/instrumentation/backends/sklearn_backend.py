"""
The scikit-learn backend
"""
from mlinspect.instrumentation.backends.backend import Backend


class SklearnBackend(Backend):
    """
    The scikit-learn backend
    """

    prefix = "sklearn"

    def before_call_used_value(self, function_info, subscript, call_code, value_code, value_value,
                               code_reference):
        """The value or module a function may be called on"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        print("sklearn_before_call_used_value")

    def before_call_used_args(self, function_info, subscript, call_code, args_code, code_reference, args_values):
        """The arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        description = None

        if function_info == ('sklearn.preprocessing._encoders', 'OneHotEncoder'):
            description = "Categorical Encoder (OneHotEncoder)"
        elif function_info == ('sklearn.preprocessing._data', 'StandardScaler'):
            description = "Numerical Encoder (StandardScaler)"
        elif function_info == ('sklearn.tree._classes', 'DecisionTreeClassifier'):
            description = "Decision Tree"

        if description:
            self.code_reference_to_description[code_reference] = description

    def before_call_used_kwargs(self, function_info, subscript, call_code, kwargs_code, code_reference, kwargs_values):
        """The keyword arguments a function may be called with"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        description = None
        if function_info == ('sklearn.preprocessing._label', 'label_binarize'):
            classes = kwargs_values['classes']
            description = "label_binarize, classes: {}".format(classes)

        if description:
            self.code_reference_to_description[code_reference] = description

    def after_call_used(self, function_info, subscript, call_code, return_value, code_reference):
        """The return value of some function"""
        # pylint: disable=too-many-arguments, unused-argument, no-self-use
        print("sklearn_after_call_used")
        return return_value
