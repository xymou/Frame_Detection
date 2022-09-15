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

class Verbalizer(nn.Module):
    r'''
    Base class for all the verbalizers. 
    Args: 
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        classes (:obj:`Sequence[str]`): A sequence of classes that need to be projected.
    '''
    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 classes: Optional[Sequence[str]] = None,
                 num_classes: Optional[int] = None,
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
            # raise AttributeError("No able to configure num_classes")
        self._in_on_label_words_set = False

    @property
    def label_words(self,):
        r'''
        Label words means the words in the vocabulary projected by the labels. 
        E.g. if we want to establish a projection in sentiment classification: positive :math:`\rightarrow` {`wonderful`, `good`},
        in this case, `wonderful` and `good` are label words.
        '''
        return self._label_words        

    @label_words.setter
    def label_words(self, label_words):
        if label_words is None:
            return
        self._label_words = self._match_label_words_to_label_ids(label_words)
        if not self._in_on_label_words_set:
            self.safe_on_label_words_set()

    def _match_label_words_to_label_ids(self, label_words): 
        """
        sort label words dict of verbalizer to match the label order of the classes
        """
        if isinstance(label_words, dict):
            if self.classes is None:
                raise ValueError("""
                classes attribute of the Verbalizer should be set since your given label words is a dict.
                Since we will match the label word with respect to class A, to A's index in classes
                """)
            if set(label_words.keys()) != set(self.classes):
                raise ValueError("name of classes in verbalizer are differnt from those of dataset")
            label_words = [ # sort the dict to match dataset
                label_words[c]
                for c in self.classes
            ] # length: label_size of the whole task
        elif isinstance(label_words, list) or isinstance(label_words, tuple):
            pass
            # logger.info("""
            # Your given label words is a list, by default, the ith label word in the list will match class i of the dataset.
            # Please make sure that they have the same order.
            # Or you can pass label words as a dict, mapping from class names to label words.
            # """)
        else:
            raise ValueError("Verbalizer label words must be list, tuple or dict")
        return label_words

    def safe_on_label_words_set(self,):
        self._in_on_label_words_set = True
        self.on_label_words_set()
        self._in_on_label_words_set = False

    def on_label_words_set(self,):
        r"""A hook to do something when textual label words were set.
        """
        pass

    @property
    def vocab(self,) -> Dict:
        if not hasattr(self, '_vocab'):
            self._vocab = self.tokenizer.convert_ids_to_tokens(np.arange(self.vocab_size).tolist())
        return self._vocab

    @property
    def vocab_size(self,) -> int:
        return self.tokenizer.vocab_size

    @abstractmethod
    def generate_parameters(self, **kwargs) -> List:
        r"""
        The verbalizer can be seen as an extra layer on top of the originial
        pre-trained models. In manual verbalizer, it is a fixed one-hot vector of dimension
        vocab_size, with the position of the label word being 1 and 0 everywhere else. 
        In other situation, the parameters may be a continuous vector over the 
        vocab, with each dimension representing a weight of that token.
        Moreover, the parameters may be set to trainable to allow label words selection.
        
        Therefore, this function serves as an abstract methods for generating the parameters
        of the verbalizer, and must be instantiated in any derived class.
        Note that the parameters need to be registered as a part of pytorch's module to 
        It can be acheived by wrapping a tensor using nn.Parameter().
        """
        raise NotImplementedError

    def register_calibrate_logits(self, logits: torch.Tensor):
        r"""
        This function aims to register logits that need to be calibrated, and detach the orginal logits from the current graph.
        """
        if logits.requires_grad:
            logits = logits.detach()
        self._calibrate_logits = logits
        
    def process_outputs(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures],
                       **kwargs):
        r"""By default, the verbalizer will process the logits of the PLM's 
        output. 
        Args:
            logits (:obj:`torch.Tensor`): The current logits generated by pre-trained language models.
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of the data.
        """

        return self.process_logits(outputs, batch=batch, **kwargs)

    def gather_outputs(self, outputs: ModelOutput):
        r""" retrieve useful output for the verbalizer from the whole model ouput
        By default, it will only retrieve the logits
        Args:
            outputs (:obj:`ModelOutput`) The output from the pretrained language model.
        Return:
            :obj:`torch.Tensor` The gathered output, should be of shape (``batch_size``, 
            ``seq_len``, ``any``)
        """
        return outputs.logits
    
    @staticmethod
    def aggregate(label_words_logits: torch.Tensor) -> torch.Tensor:
        r""" To aggregate logits on multiple label words into the label's logits
        Basic aggregator: mean of each label words' logits to a label's logits
        Can be re-implemented in advanced verbaliezer.
        Args:
            label_words_logits (:obj:`torch.Tensor`): The logits of the label words only.
        Return:
            :obj:`torch.Tensor`: The final logits calculated by the label words.
        """
        if label_words_logits.dim()>2:
            return label_words_logits.mean(dim=-1)
        else:
            return label_words_logits


    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        r"""
        Given logits regarding the entire vocab, calculate the probs over the label words set by softmax.
       
        Args:
            logits(:obj:`Tensor`): The logits of the entire vocab.
        Returns:
            :obj:`Tensor`: The probability distribution over the label words set.
        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    @abstractmethod
    def project(self,
                logits: torch.Tensor,
                **kwargs) -> torch.Tensor:
        r"""This method receives input logits of shape (batch_size, vocab_size), and use the 
        parameters of this verbalizer to project the logits over entire vocab into the
        logits of labels words. 
        Args: 
            logits (:obj:`Tensor`): The logits over entire vocab generated by the pre-trained lanuage model with shape [``batch_size``, ``max_seq_length``, ``vocab_size``] 
        
        Returns:
            :obj:`Tensor`: The normalized probs (sum to 1) of each label .
        """
        raise NotImplementedError
    
    def handle_multi_token(self, label_words_logits, mask):
        r"""
        Support multiple methods to handle the multi tokens produced by the tokenizer.
        We suggest using 'first' or 'max' if the some parts of the tokenization is not meaningful.
        Can broadcast to 3-d tensor.
    
        Args:
            label_words_logits (:obj:`torch.Tensor`):
        
        Returns:
            :obj:`torch.Tensor`
        """
        if self.multi_token_handler == "first":
            label_words_logits = label_words_logits.select(dim=-1, index=0)
        elif self.multi_token_handler == "max":
            label_words_logits = label_words_logits - 1000*(1-mask.unsqueeze(0))
            label_words_logits = label_words_logits.max(dim=-1).values
        elif self.multi_token_handler == "mean":
            label_words_logits = (label_words_logits*mask.unsqueeze(0)).sum(dim=-1)/(mask.unsqueeze(0).sum(dim=-1)+1e-15)
        else:
            raise ValueError("multi_token_handler {} not configured".format(self.multi_token_handler))
        return label_words_logits
    
    @classmethod
    def from_config(cls, 
                    config: CfgNode, 
                    **kwargs):
        r"""load a verbalizer from verbalizer's configuration node. 
        Args:
            config (:obj:`CfgNode`): the sub-configuration of verbalizer, i.e. config[config.verbalizer]
                        if config is a global config node. 
            kwargs: Other kwargs that might be used in initialize the verbalizer. 
                    The actual value should match the arguments of __init__ functions.
        """

        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs} if config is not None else kwargs
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        verbalizer = cls(**init_dict)
        if hasattr(verbalizer, "from_file"):
            if not hasattr(config, "file_path"):
                pass
            else:
                if (not hasattr(config, "label_words") or config.label_words is None) and config.file_path is not None:
                    if config.choice is None:
                        config.choice = 0
                    verbalizer.from_file(config.file_path, config.choice)
                elif (hasattr(config, "label_words") and config.label_words is not None) and config.file_path is not None:
                    raise RuntimeError("The text can't be both set from `text` and `file_path`.")
        return verbalizer
    
    def from_file(self,
                  path: str, 
                  choice: Optional[int] = 0 ):
        r"""Load the predefined label words from verbalizer file.
        Currently support three types of file format:
        1. a .jsonl or .json file, in which is a single verbalizer 
        in dict format.
        2. a .jsonal or .json file, in which is a list of verbalizers in dict format
        3.  a .txt or a .csv file, in which is the label words of a class are listed in line, 
        seperated by commas. Begin a new verbalizer by an empty line.
        This format is recommended when you don't know the name of each class.
        The details of verbalizer format can be seen in :ref:`How_to_write_a_verbalizer`. 
        Args: 
            path (:obj:`str`): The path of the local template file.
            choice (:obj:`int`): The choice of verbalizer in a file containing
                             multiple verbalizers.
        
        Returns:
            Template : `self` object
        """
        if path.endswith(".txt") or path.endswith(".csv"):
            with open(path, 'r') as f:
                lines = f.readlines()
                label_words_all = []
                label_words_single_group = []
                for line in lines:
                    line = line.strip().strip(" ")
                    if line == "":
                        if len(label_words_single_group)>0:
                            label_words_all.append(label_words_single_group)
                        label_words_single_group = []
                    else:
                        label_words_single_group.append(line)
                if len(label_words_single_group) > 0: # if no empty line in the last
                    label_words_all.append(label_words_single_group)
                if choice >= len(label_words_all):
                    raise RuntimeError("choice {} exceed the number of verbalizers {}"
                                .format(choice, len(label_words_all)))

                label_words = label_words_all[choice]
                label_words = [label_words_per_label.strip().split(",") \
                            for label_words_per_label in label_words]
            
        elif path.endswith(".jsonl") or path.endswith(".json"):
            with open(path, "r") as f:
                label_words_all = json.load(f)
                # if it is a file containing multiple verbalizers
                if isinstance(label_words_all, list):
                    if choice >= len(label_words_all):
                        raise RuntimeError("choice {} exceed the number of verbalizers {}"
                                .format(choice, len(label_words_all)))
                    label_words = label_words_all[choice]
                elif isinstance(label_words_all, dict):
                    label_words = label_words_all
                    if choice>0:
                        logger.warning("Choice of verbalizer is 1, but the file  \
                        only contains one verbalizer.")
                
        self.label_words = label_words
        if self.num_classes is not None:
            num_classes = len(self.label_words)
            assert num_classes==self.num_classes, 'number of classes in the verbalizer file\
                                            does not match the predefined num_classes.'
        return self        