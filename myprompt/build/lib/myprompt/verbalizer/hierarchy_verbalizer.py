import json
from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode
from myprompt.data.example import InputFeatures
import re
import numpy as np
from myprompt.verbalizer.base import Verbalizer
from myprompt.verbalizer.manual_verbalizer import ManualVerbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from myprompt.utils.logging import logger

"""
1.加权concept的prob
"""
class WeightedVerbalizer(ManualVerbalizer):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 concepts: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                ):
        super().__init__(
                 tokenizer=tokenizer,
                 classes=classes,
                 num_classes=num_classes,
                 label_words=concepts,
                 prefix=prefix,
                 multi_token_handler=multi_token_handler,
                 post_log_softmax=post_log_softmax,
        )
        self.weight = nn.Parameter(torch.zeros_like(self.label_words_mask, dtype=torch.float32), requires_grad = True)

    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        weight = F.softmax(self.weight-10000*(1-self.label_words_mask), dim = 1)
        label_words_logits = (label_words_logits * self.label_words_mask * weight).sum(-1)
        return label_words_logits


"""
2.使用parent class 的concepts初始化虚拟答案
直接调adaptive verbalizer即可
"""

"""
3.假定每个DF有n个ideal words,
把带mask的GF*V1矩阵，变为DF * V2矩阵
(分两种实现: 1.得到GF * V2后考虑全连接，即DF与所有GF全连接；
2.得到GF * V2后只考虑对parent class的加起来，实现上应该多一个mask矩阵)
"""
def _init_fc_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)  #Linear层的参数
        if module.bias is not None:
            nn.init.constant_(module.bias.data,0.0) 
class TransVerbalizer(Verbalizer):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 target_classes: Optional[List] = None,
                 target_words_num:Optional[int] = 10,
                 parent_classes: Optional[Dict] =None,
                 mask_non_parent: Optional[bool] = False,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.target_classes = target_classes
        self.target_words_num = target_words_num
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax
        self.fc1 = nn.Linear(self.count_max_num_label_words(label_words), target_words_num) #V1 * V2
        self.fc2 = nn.Linear(len(classes), len(target_classes))  # GF * DF
        self.apply(_init_fc_weights)
        if mask_non_parent:
            self.parent_mask_matrix = self.generate_mask_matrix(classes, target_classes, parent_classes)
        else:
            self.parent_mask_matrix = torch.ones((len(classes), len(target_classes)))

    def generate_mask_matrix(self, classes, target_classes, parent_classes):
        matrix = np.zeros((len(classes), len(target_classes)))
        for i in range(len(classes)):
            for j in range(len(target_classes)):
                if classes[i] in parent_classes[target_classes[j]]:
                    matrix[i][j]=1
        return torch.tensor(matrix, dtype=torch.float32)

    def count_max_num_label_words(self, label_words):    
        if isinstance(label_words, dict):    
            max_num_label_words = max([len(label_words[key]) for key in label_words])
        elif isinstance(label_words, list):
            max_num_label_words = max([len(key) for key in label_words])
        else:
            raise ValueError('Need Dict or List for label words.')
        return max_num_label_words

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

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        
    
    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)
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

    def project(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        # print(0,label_words_logits.size())
        # print(1,self.fc1(label_words_logits).size())
        # print(2,self.fc2(self.fc1(label_words_logits).transpose(1,2)).size())
        # print(3,self.parent_mask_matrix.size())
        # print(4,self.fc2.weight.T.size())
        self.parent_mask_matrix =self.parent_mask_matrix.to(logits.device)
        label_words_logits = (self.fc1(label_words_logits).transpose(1,2).matmul(self.fc2.weight.T*self.parent_mask_matrix)).transpose(1,2)
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
        label_words_logits = (label_words_logits ).mean(-1)
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
4.复现Concept-based 的思路，好像套不了，
原文最终是得到标签的表示，和文本表示融合作分类
但是在PMT框架里输出的是[MASK] 在vocab子集上的概率;这个子集要么来自于已有的词(parent concepts),
要么是添加的新词，可是添加的新词在训练过程中就和其他词一样了只受MLM影响，如果用参数加权各个concepts的权重参数并不能得到更新


"""
class HierarchyVerbalizer(nn.Module):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 parent_classes: Optional[Mapping[str, List]] = None,
                 parent_concepts: Optional[Mapping[str, List]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                ):
        super().__init__()
        self.tokenizer = tokenizer
        self.classes = classes
        if classes is not None and num_classes is not None:
            assert len(classes) == num_classes, "len(classes) != num_classes, Check you config."
            self.num_classes = num_classes
        elif num_classes is not None:
            self.num_classes = num_classes
        elif classes is not None:
            self.num_classes = len(classes)
        else:
            self.num_classes = None 
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.parent_classes = classes 
        self.parent_concepts = parent_concepts
        self.post_log_softmax = post_log_softmax  
        self.concepts = self._match_concepts_from_parent_classes()  


    def _match_concepts_from_parent_classes(self,):
        concepts = []
        if self.parent_classes is None or self.parent_concepts is None:
            raise ValueError("""
                            Must provide classes and concepts of parent classes.
                            """)
        for c in self.classes:
            tmp = []
            parent_lst = self.parent_classes[c]
            for p in parent_lst:
                tmp += self.parent_concepts[p]
            concepts.append(list(set(tmp)))
        return concepts