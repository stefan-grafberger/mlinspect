"""
Monkey patching for sklearn
"""

import gorilla
import numpy
from sklearn import preprocessing, compose, tree, impute, linear_model, model_selection
from example_pipelines.healthcare import healthcare_utils

from mlinspect.backends._backend import BackendResult
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import OperatorContext
from mlinspect.instrumentation import _pipeline_executor
from mlinspect.instrumentation._dag_node import OperatorType, DagNode
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching.monkey_patching_utils import execute_patched_func, add_dag_node, \
    execute_patched_func_indirect_allowed, get_input_info, execute_patched_func_no_op_id
from mlinspect.monkeypatching.numpy import MlinspectNdarray


@gorilla.patches(healthcare_utils.MyW2VTransformer)
class SklearnMyW2VTransformerPatching:
    """ Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                        workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5,
                        null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None):
        """ Patch for ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(healthcare_utils.MyW2VTransformer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, size=size, alpha=alpha, window=window,
                     min_count=min_count, max_vocab_size=max_vocab_size, sample=sample,
                     seed=seed, workers=workers, min_alpha=min_alpha, sg=sg, hs=hs,
                     negative=negative, cbow_mean=cbow_mean, hashfxn=hashfxn, iter=iter,
                     null_word=null_word, trim_rule=trim_rule, sorted_vocab=sorted_vocab,
                     batch_words=batch_words)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, size=size, alpha=alpha, window=window,
                                             min_count=min_count, max_vocab_size=max_vocab_size, sample=sample,
                                             seed=seed, workers=workers, min_alpha=min_alpha, sg=sg, hs=hs,
                                             negative=negative, cbow_mean=cbow_mean, hashfxn=hashfxn, iter=iter,
                                             null_word=null_word, trim_rule=trim_rule, sorted_vocab=sorted_vocab,
                                             batch_words=batch_words)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('example_pipelines.healthcare.healthcare_utils.MyW2VTransformer', 'fit_transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(healthcare_utils.MyW2VTransformer, 'fit_transform')
        function_info = ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result)
        new_return_value = backend_result.annotated_dfobject.result_data
        assert isinstance(new_return_value, MlinspectNdarray)
        dag_node = DagNode(singleton.get_next_op_id(), self.mlinspect_caller_filename, self.mlinspect_lineno,
                           OperatorType.TRANSFORMER, function_info, "Word2Vec", ['array'],
                           self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        return new_return_value
