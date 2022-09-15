from abc import abstractmethod
import json
from transformers.file_utils import ModelOutput
from myprompt.config import convert_cfg_to_dict
from transformers.utils.dummy_pt_objects import PreTrainedModel
from myprompt.utils.utils import signature

from yacs.config import CfgNode
from myprompt.data.example import InputFeatures, InputExample
import torch
import torch.nn as nn
from typing import *
from transformers.tokenization_utils import PreTrainedTokenizer

from myprompt.utils.logging import logger
import numpy as np
import torch.nn.functional as F

class Template(nn.Module):
    r'''
    Base class for all the templates. 
    Most of methods are abstract, with some expections to hold the common methods for all template, such as ``loss_ids``, ``save``, ``load``.
    Args: 
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        mask_token (:obj:`str`): The special token that is masked and need to be predicted by the model.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. 
    '''
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]
    def __init__(self,
                tokenizer: PreTrainedTokenizer,
                mask_token: str = '<mask>',
                placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'}):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.placeholder_mapping = placeholder_mapping
        self._in_on_text_set = False

        self.mixed_token_start = "{"
        self.mixed_token_end = "}"

    def get_default_loss_ids(self) -> List[int]:
        """
        get the loss indices for the template using mask
        1 for masked tokens
        0 for sequence tokens
        """           
        return [1 if 'mask' in d else 0 for d in self.text]

    def get_default_shortenable_ids(self) -> List[int]:
        """
        get shortenable ids, denoting which part of the template can be truncated to fit the LM's max_seq_leng
        default: the input text is shortenable, while the template text and other special tokens are not shortenable
        1 for input tokens
        0 for template sequence tokens
        """
        idx = []
        for d in self.text:
            if 'shortenable' in d:
                idx.append(1 if d['shortable'] else 0)
            else:
                idx.append(1 if 'placeholder' in d else 0)
        return idx
    
    def get_default_soft_token_ids(self) -> List[int]:
        r'''
        This function identifies which tokens are soft tokens.
        Sometimes tokens in the template are not from the vocabulary, 
        but a sequence of soft tokens.
        In this case, you need to implement this function
        Raises:
            NotImplementedError: if needed, add ``soft_token_ids`` into ``registered_inputflag_names`` attribute of Template class and implement this method.
        '''
        raise NotImplementedError

    def incorporate_text_example(self,
                                 example: InputExample
                                ) -> List[str]:
        """Given an example, replace placeholder of text_a, text_b and meta information by real data
        Args:
            example (:obj:`InputExample`): An InputExample object, which should have attributes that are able to be filled in the template.
        Returns:
            List[str]: a list of str of the same length as self.text. the placeholder and meta information are replaced by real data information.
        """
        text = self.text.copy()
        for placeholder_token in self.placeholder_mapping:
            for i in range(len(text)):
                text[i] = " " + text[i].replace(placeholder_token, getattr(example, self.placeholder_mapping[placeholder_token]))
        for key, value in example.meta.items():
            for i in range(len(text)):
                text[i] = " " + text[i].replace("<meta:"+key+">", value)
        return text        

    def incorporate_text_example(self,
                                 example: InputExample
                                ):
        text = self.text.copy()
        for i, d in enumerate(text):
            if 'placeholder' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(getattr(example, d['placeholder']))
            elif 'meta' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(example.meta[d['meta']])
            elif 'soft' in d:
                text[i] = ''; # unused
            elif 'mask' in d:
                text[i] = '<mask>'
            elif 'special' in d:
                text[i] = d['special']
            elif 'text' in d:
                text[i] = d["add_prefix_space"] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        return text

    def _check_template_format(self, ):
        r"""check whether the template format is correct.
        """
        mask_num = 0
        for i, d in enumerate(self.text):
            if 'mask' in d:
                mask_num += 1
        
        if mask_num==0:
            raise RuntimeError(f"'mask' position not found in the template: {self.text}. Please Check!")

    def parse_text(self, text: str) -> List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {"add_prefix_space":' ' if (i>0 and text[i-1] == ' ') else ''}
            while i < len(text) and text[i] == ' ':
                d["add_prefix_space"]=' '
                i+=1
            if i==len(text):break
        
            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d['text']=text[i:j].rstrip(' ')
                i = j
            else:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        break
                    j = j + 1
                if j == len(text):
                    raise ValueError(f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")
                dict_str = '{'+text[i+1:j]+'}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    import traceback
                    print(traceback.format_exc())
                    print(f"syntax error in {dict_str}")
                    exit()
                i = j + 1
            parsed.append(d)
        return parsed                    

    def wrap_one_example(self, 
                         example: InputExample) -> List[Dict]:
        r'''Given an input example which contains input text, which can be referenced
        by self.template.placeholder_mapping 's value. 
        This function process the example into a list of dict,
        Each dict functions as a group, which has the sample properties, such as
        whether it's shortenable, whether it's the masked position, whether it's soft token, etc.
        Since a text will be tokenized in the subsequent processing procedure,
        these attributes are broadcasted along the tokenized sentence.
        
        Args:
            example (:obj:`InputExample`): An InputExample object, which should have attributes that are able to be filled in the template.
       
        Returns:
            :obj:`List[Dict]` a list of dict of the same length as self.text. e.g. [{"loss_ids": 0, "text": "It was"}, {"loss_ids": 1, "text": "<mask>"}, ]
        '''
        if self.text is None:
            raise ValueError("template text has not been initialized")
        if isinstance(example, InputExample):
            text = self.incorporate_text_example(example)

            not_empty_keys = example.keys()
            for placeholder_token in self.placeholder_mapping:
                if self.placeholder_mapping[placeholder_token] in not_empty_keys:
                    not_empty_keys.remove(self.placeholder_mapping[placeholder_token]) # placeholder has been processed, remove
            not_empty_keys.remove('meta') # meta has been processed

            keys, values= ['text'], [text]
            for inputflag_name in self.registered_inputflag_names:
                keys.append(inputflag_name)
                v = None
                if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:
                    v = getattr(self, inputflag_name)
                elif hasattr(self, "get_default_"+inputflag_name):
                    v = getattr(self, "get_default_"+inputflag_name)()
                    setattr(self, inputflag_name, v) # cache 
                else:
                    raise ValueError("""
                    Template's inputflag '{}' is registered but not initialize.
                    Try using template.{} = [...] to initialize
                    or create an method get_default_{}(self) in your template.
                    """.format(inputflag_name, inputflag_name, inputflag_name))
                
                if len(v) != len(text):
                    raise ValueError("Template: len({})={} doesn't match len(text)={}."\
                        .format(inputflag_name, len(v), len(text)))
                values.append(v)
            wrapped_parts_to_tokenize = []
            for piece in list(zip(*values)):
                wrapped_parts_to_tokenize.append(dict(zip(keys, piece)))

            wrapped_parts_not_tokenize = {key: getattr(example, key) for key in not_empty_keys}
            return [wrapped_parts_to_tokenize, wrapped_parts_not_tokenize]
        else:
            raise TypeError("InputExample")                     

    @abstractmethod
    def process_batch(self, batch):
        r"""Template should rewrite this method if you need to process the batch input such as substituting embeddings.
        """
        return batch # not being processed

    def post_processing_outputs(self, outputs):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        return outputs

    def save(self,
             path: str,
             **kwargs) -> None:
        raise NotImplementedError

    @property
    def text(self):
        return self._text

    @text.setter 
    def text(self, text):
        self._text = text
        if text is None:
            return
        if not self._in_on_text_set:
            self.safe_on_text_set()
        self._check_template_format()        

    def safe_on_text_set(self) -> None:
        r"""With this wrapper function, setting text inside ``on_text_set()``
            will not trigger ``on_text_set()`` again to prevent endless recursion.
        """
        self._in_on_text_set = True
        self.on_text_set()
        self._in_on_text_set = False

    @abstractmethod
    def on_text_set(self):
        r"""
        A hook to do something when template text was set.
        The designer of the template should explictly know what should be down when the template text is set.
        """
        raise NotImplementedError

    def from_file(self,
                  path: str,
                  choice: int = 0,
                 ):
        r'''
        Read the template from a local file.
        Args: 
            path (:obj:`str`): The path of the local template file.
            choice (:obj:`int`): The id-th line of the file.
        '''
        with open(path, 'r') as fin:
            text = fin.readlines()[choice].rstrip()
            logger.info(f"using template: {text}")
        self.text = text
        return self

    @classmethod
    def from_config(cls,
                    config: CfgNode,
                    **kwargs):
        r"""load a template from template's configuration node. 
        Args:
            config (:obj:`CfgNode`): the sub-configuration of template, i.e. config[config.template]
                        if config is a global config node. 
            kwargs: Other kwargs that might be used in initialize the verbalizer. 
                    The actual value should match the arguments of __init__ functions.
        """

        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        template = cls(**init_dict)
        if hasattr(template, "from_file"):
            if not hasattr(config, "file_path"):
                pass
            else:
                if (not hasattr(config, "text") or config.text is None) and config.file_path is not None:
                    if config.choice is None:
                        config.choice = 0
                    template.from_file(config.file_path, config.choice)
                elif (hasattr(config, "text") and config.text is not None) and config.file_path is not None:
                    raise RuntimeError("The text can't be both set from `text` and `file_path`.")
        return template       