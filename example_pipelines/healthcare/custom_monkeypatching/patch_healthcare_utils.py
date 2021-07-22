"""
Monkey patching for healthcare_utils
"""
import gorilla
from gensim import sklearn_api

from example_pipelines.healthcare import healthcare_utils
from mlinspect.backends._sklearn_backend import SklearnBackend
from mlinspect.inspections._inspection_input import OperatorContext, FunctionInfo, OperatorType
from mlinspect.instrumentation._dag_node import DagNode, BasicCodeLocation, DagNodeDetails
from mlinspect.instrumentation._pipeline_executor import singleton
from mlinspect.monkeypatching._monkey_patching_utils import add_dag_node, \
    get_input_info, execute_patched_func_no_op_id, get_optional_code_info_or_none
from mlinspect.monkeypatching._mlinspect_ndarray import MlinspectNdarray


class SklearnMyW2VTransformerPatching:
    """ Patches for healthcare_utils.MyW2VTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.patch(sklearn_api.W2VTransformer, name='__init__', settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, *, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                        workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5,
                        null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False):
        """ Patch for ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, redefined-builtin,
        # pylint: disable=invalid-name
        original = gorilla.get_original_attribute(sklearn_api.W2VTransformer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {'size': size, 'alpha': alpha, 'window': window,
                                             'min_count': min_count, 'max_vocab_size': max_vocab_size, 'sample': sample,
                                             'seed': seed, 'workers': workers, 'min_alpha': min_alpha, 'sg': sg,
                                             'hs': hs, 'negative': negative, 'cbow_mean': cbow_mean, 'iter': iter,
                                             'null_word': null_word, 'trim_rule': trim_rule,
                                             'sorted_vocab': sorted_vocab, 'batch_words': batch_words}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, hashfxn=hashfxn, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, hashfxn=hashfxn,
                                             **self.mlinspect_non_data_func_args)

    @gorilla.patch(healthcare_utils.MyW2VTransformer, name='fit_transform', settings=gorilla.Settings(allow_hit=True))
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('example_pipelines.healthcare.healthcare_utils.MyW2VTransformer', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(healthcare_utils.MyW2VTransformer, 'fit_transform')
        function_info = FunctionInfo('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
        result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
        backend_result = SklearnBackend.after_call(operator_context,
                                                   input_infos,
                                                   result,
                                                   self.mlinspect_non_data_func_args)
        new_return_value = backend_result.annotated_dfobject.result_data
        assert isinstance(new_return_value, MlinspectNdarray)
        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Word2Vec: fit_transform", ['array']),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        add_dag_node(dag_node, [input_info.dag_node], backend_result)
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_return_value

    @gorilla.patch(healthcare_utils.MyW2VTransformer, name='transform', settings=gorilla.Settings(allow_hit=True))
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('example_pipelines.healthcare.healthcare_utils.MyW2VTransformer', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(healthcare_utils.MyW2VTransformer, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            input_infos = SklearnBackend.before_call(operator_context, [input_info.annotated_dfobject])
            result = original(self, input_infos[0].result_data, *args[1:], **kwargs)
            backend_result = SklearnBackend.after_call(operator_context,
                                                       input_infos,
                                                       result,
                                                       self.mlinspect_non_data_func_args)
            new_return_value = backend_result.annotated_dfobject.result_data
            assert isinstance(new_return_value, MlinspectNdarray)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Word2Vec: transform", ['array']),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code))
            add_dag_node(dag_node, [input_info.dag_node], backend_result)
        else:
            new_return_value = original(self, *args, **kwargs)
        return new_return_value
