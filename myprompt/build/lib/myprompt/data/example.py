import copy
import json
import pickle
from typing import *
from myprompt.utils.logging import logger

import torch
from torch.utils.data._utils.collate import default_collate

class InputExample(object):
    """
    A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.
    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always neccessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """
    def __init__(self,
                guid = None,
                text_a = "",
                text_b = "",
                label = None,
                meta: Optional[Dict] = None,
                tgt_text: Optional[Union[str,List[str]]] = None,
                issue = None,
                ):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text
        self.issue = issue

    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"        

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path:str) -> List['InputExample']:
        with open(path ,'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path:str) -> None:
        with open(path,'wb') as fh:
            pickle.dump(examples, fh)
    
class InputFeatures(dict):
    """
    The class for input to the PLM and Prompts. To make users explicitly know the available keys, 
    we define a dict with a set of predefined possible keys. The default value to any key is None.
    When use it as a dict, all the keys whose values are None are invisible.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems. 
    """
    tensorable_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
                    'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids',
                    'past_key_values', 'loss_ids','issue','mlm_loss_ids','mlm_labels']
    all_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label',
        'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids', 
        'past_key_values', 'loss_ids', 'guid', 'tgt_text', 'encoded_tgt_text', 'input_ids_len','issue','mlm_loss_ids','mlm_labels']
    non_tensorable_keys = []   
    
    def __init__(self,
                input_ids: Optional[Union[List, torch.Tensor]] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[Union[List[int], torch.Tensor]] = None,
                token_type_ids: Optional[Union[List[int], torch.Tensor]] = None,
                label: Optional[Union[int, torch.Tensor]] = None,
                decoder_input_ids: Optional[Union[List, torch.Tensor]] = None,
                decoder_inputs_embeds: Optional[torch.Tensor] = None,
                soft_token_ids: Optional[Union[List, torch.Tensor]] = None,
                past_key_values: Optional[torch.Tensor] = None,  # for prefix_tuning
                loss_ids: Optional[Union[List, torch.Tensor]] = None,
                guid: Optional[str] = None,
                tgt_text: Optional[str] = None,
                use_cache: Optional[bool] = None,
                encoded_tgt_text: Optional[str] = None,
                input_ids_len: Optional[int] = None,
                issue:Optional[int] = None,
                mlm_loss_ids:Optional[Union[List, torch.Tensor]] = None,
                mlm_labels:Optional[Union[List, torch.Tensor]] = None,
                **kwargs):
        self.input_ids = input_ids
        self.inputs_embeds = inputs_embeds
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids
        self.decoder_inputs_embeds = decoder_inputs_embeds
        self.soft_token_ids = soft_token_ids
        self.past_key_values = past_key_values
        self.loss_ids = loss_ids
        self.guid = guid
        self.tgt_text = tgt_text
        self.encoded_tgt_text = encoded_tgt_text
        self.use_cache = use_cache
        self.input_ids_len = input_ids_len
        self.issue = issue
        self.mlm_loss_ids=mlm_loss_ids
        self.mlm_labels=mlm_labels

        for k in kwargs.keys():
            logger.warning("Your are passing an unexpected key words: {} to InputFeatures, might yield unexpected behaviours!".format(k))
            setattr(self, k, kwargs[k])

    @classmethod
    def add_tensorable_keys(cls, *args):
        cls.tensorable_keys.extend(args)
    
    @classmethod
    def add_not_tensorable_keys(cls, *args):
        cls.not_tensorable_keys.extend(args)

    @classmethod
    def add_keys(cls, *args):
        cls.all_keys.extend(args)

    def __repr__(self):
        return str(self.to_json_string())
    
    def __len__(self):
        return len(self.keys())

    def to_tensor(self):
        """inplace operation, convert all tensorable features into :obj:`torch.tensor`"""
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, torch.tensor(value))
        return self                

    def to(self, device: str = "cuda:0"):
        r"""move the tensor keys to runtime device, such as gpu:0
        """

        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, value.to(device))
        return self

    def cuda(self):
        r"""mimic the tensor behavior
        """
        return self.to()

    def to_json_string(self, keep_none=False):
        """Serializes this instance to a JSON string."""
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):   
                data[key] =  value.detach().cpu().tolist()
            elif value is None and keep_none:
                data[key] = None
            else:
                data[key] = value
        return json.dumps(data) + "\n"

    def keys(self, keep_none=False) -> List[str]:
        """get all keys of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`List[str]`: keys of the InputFeatures
        """
        if keep_none:
            return self.all_keys
        else:
            return [key for key in self.all_keys if getattr(self, key) is not None]

    def to_dict(self, keep_none=False) -> Dict[str, Any]:
        """get the dict of mapping from keys to values of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`Dict[str, Any]`: dict of mapping from keys to values of the InputFeatures
        """
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if value is not None:
                data[key] =  value
            elif value is None and keep_none:
                data[key] = None
        return data        

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __iter__(self):
        return iter(self.keys())
    
    def __setitem__(self, key, item):
        if key not in self.all_keys:
            raise KeyError("Key {} not in predefined set of keys".format(key))
        setattr(self, key, item)

    def values(self, keep_none=False) -> List[Any]:
        """get the values with respect to the keys  of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`List[Any]`: the values with respect to the keys of the InputFeatures
        """
        return [getattr(self, key) for key in self.keys(keep_none=keep_none)]

    def __contains__(self, key, keep_none=False):
        return key in self.keys(keep_none)
    
    def items(self,):
        """get the (key, value) pairs  of the InputFeatures
        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.
        Returns:
            :obj:`List[Any]`: the (key, value) pairs of the InputFeatures
        """
        return [(key, self.__getitem__(key)) for key in self.keys()]

    @staticmethod
    def collate_fct(batch: List):
        r'''
        This function is used to collate the input_features.
        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.
        Returns:
            :obj:`InputFeatures`: Return the InputFeatures of the current batch of data.
        '''

        elem = batch[0]
        return_dict = {}
        for key in elem:
            if key == "encoded_tgt_text":
                return_dict[key] = [d[key] for d in batch]
            else:
                return_dict[key] = default_collate([d[key] for d in batch])
        return InputFeatures(**return_dict)        



 