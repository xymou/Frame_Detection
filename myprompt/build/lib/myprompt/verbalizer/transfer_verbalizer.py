"""
1. C1 * V1 的分布transfer到 C2 * V2的分布
"""

import json
from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode
from myprompt.data.example import InputFeatures
import re
from myprompt.verbalizer.base import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from myprompt.utils.logging import logger


def _init_fc_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)  #Linear层的参数
        if module.bias is not None:
            nn.init.constant_(module.bias.data,0.0) 
            
class TransferVerbalizer(Verbalizer):
    r"""
    TODO
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 target_classes: Optional[List] = None,
                 target_label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.target_classes = target_classes
        self.target_label_words = target_label_words
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax
        self.fc1 = nn.Linear(self.count_max_num_label_words(label_words), self.count_max_num_label_words(target_label_words))
        self.fc2 = nn.Linear(len(label_words), len(target_label_words))
        self.apply(_init_fc_weights)
        self._in_on_target_label_words_set = False

    def count_max_num_label_words(self, label_words):    
        if isinstance(label_words, dict):    
            max_num_label_words = max([len(label_words[key]) for key in label_words])
        elif isinstance(label_words, list):
            max_num_label_words = max([len(key) for key in label_words])
        else:
            raise ValueError('Need Dict or List for label words.')
        return max_num_label_words

    @property
    def target_label_words(self,):
        r'''
        Label words means the words in the vocabulary projected by the labels. 
        E.g. if we want to establish a projection in sentiment classification: positive :math:`\rightarrow` {`wonderful`, `good`},
        in this case, `wonderful` and `good` are label words.
        '''
        return self._target_label_words        

    @target_label_words.setter
    def target_label_words(self, target_label_words):
        if target_label_words is None:
            return
        self._target_label_words = self._match_target_label_words_to_label_ids(target_label_words)

    def _match_target_label_words_to_label_ids(self, target_label_words): 
        if isinstance(target_label_words, dict):
            if self.target_classes is None:
                raise ValueError("""
                classes attribute of the Verbalizer should be set since your given label words is a dict.
                Since we will match the label word with respect to class A, to A's index in classes
                """)
            if set(target_label_words.keys()) != set(self.target_classes):
                raise ValueError("name of classes in verbalizer are differnt from those of dataset")
            target_label_words = [
                target_label_words[c]
                for c in self.target_classes
            ] 
            
        elif isinstance(target_label_words, list) or isinstance(target_label_words, tuple):
            pass
        else:
            raise ValueError("Verbalizer label words must be list, tuple or dict")
        return target_label_words

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.target_label_words = self.add_prefix(self.target_label_words, self.prefix)

        self.generate_parameters()
        
    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.
        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.
        
        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words
        
    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token. 
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        
        all_ids = []
        for words_per_label in self.target_label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)        
        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        target_words_ids_mask = torch.zeros(max_num_label_words, max_len)
        target_words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        target_words_id_mask = torch.tensor(target_words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        self.target_label_words_mask = nn.Parameter(torch.clamp(target_words_id_mask.sum(dim=-1), max=1), requires_grad=False)

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words. 
        
        Args:
            logits (:obj:`torch.Tensor`): The orginal logits of label words.
        
        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        # print(0,label_words_logits.size())
        # print(1,self.fc1(label_words_logits).transpose(1,2).size())
        # print(2,self.fc2(self.fc1(label_words_logits).transpose(1,2)).transpose(1,2).size())
        label_words_logits = self.fc2(self.fc1(label_words_logits).transpose(1,2)).transpose(1,2)
        #label_words_logits -= 10000*(1-self.label_words_mask)
        label_words_logits -= 10000*(1-self.target_label_words_mask)
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps: 
        (1) Project the logits into logits of label words
        if self.post_log_softmax is True:
            (2) Normalize over all label words
            (3) Calibrate (optional)
        (4) Aggregate (for multiple label words)
        Args:
            logits (:obj:`torch.Tensor`): The orginal logits.
        
        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)
        
        
        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        # aggreate
        label_logits = self.aggregate(label_words_logits)
        return label_logits
    
    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.
        
        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.
        Returns:
            :obj:`Tensor`: The logits over the label words set.
        
        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)


    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.
        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.
        
        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words. 
        """
        #label_words_logits = (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1)
        label_words_logits = (label_words_logits * self.target_label_words_mask).sum(-1)/self.target_label_words_mask.sum(-1)
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        
        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]
        
        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() ==  1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,keepdim=True) # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

"""
2. 把DF的(vocab)分布视作各个GF的(vocab)分布的加权；可是两者的vocab不一样，怎么做
    某个GF的(vocab)怎么获得？主题模型？
"""