"""
把MLM和目标任务（分类）放在一起训
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import *
from myprompt.data.example import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from torch.utils.data import DataLoader
from myprompt.utils.logging import logger
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from myprompt.template import Template,PtuningTemplate
from myprompt.verbalizer import Verbalizer
from myprompt.plm.utils import TokenizerWrapper
from collections import defaultdict
from myprompt.utils.logging import logger
from myprompt.utils.utils import round_list, signature
from transformers import  AdamW, get_linear_schedule_with_warmup
from myprompt.model.base import PromptModel
import copy
from myprompt.utils.utils import ReverseLayerF

def _init_fc_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)  #Linear层的参数
        if module.bias is not None:
            nn.init.constant_(module.bias.data,0.0) 

class JointMLM(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 verbalizer_label:Verbalizer,
                 verbalizer_issue:Verbalizer,
                 #label_num: int,
                 hidden_size:int,
                 pivot_num:int,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.verbalizer_label = verbalizer_label
        self.verbalizer_issue = verbalizer_issue
        self.keys = list(template.keys())
        self.pivot_num = pivot_num
        self.fc = nn.Linear(hidden_size, pivot_num)
        #self.label_emb = nn.Embedding(label_num, hidden_size) #label embedding 可学习的参数，代替token embeddings
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
        self.apply(_init_fc_weights)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha, train=True) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch, out1 = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch),diff=True)
        share_batch, out2 = self.template_share.process_batch(copy.deepcopy(batch),diff=True)
        
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        share_batch['inputs_embeds'] = ReverseLayerF.apply(share_batch['inputs_embeds'], alpha)
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        issue_outputs = self.plm(**share_batch, output_hidden_states=True, output_attentions = True)[0]  #mlm output
        issue_outputs = self.template_share.post_processing_outputs(issue_outputs)  
        if train:
            #mlm 应该用哪个输入，用share batch吧，其实感觉用share_batch 更好？希望填对这些pivot是与issue无关的？
            mlm_outputs = self.plm(**share_batch)[0] #batch_size * seq_len *hidden_size
            mlm_logits = self.fc(mlm_outputs)
            mlm_logits = mlm_logits.view(-1, self.pivot_num)

        label_outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)
        issue_outputs = self.verbalizer_issue.gather_outputs(issue_outputs)
        outputs_at_mask = self.extract_at_mask(issue_outputs, batch)
        issue_logits = self.verbalizer_issue.process_logits2(outputs_at_mask, batch=batch)
        if train:
            return label_words_logits, issue_logits, mlm_logits, out1, out2
        else:
            return label_words_logits, issue_logits, None, out1, out2
    
    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device
    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):

        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer_label.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        _state_dict['verbalizer_issue'] = self.verbalizer_issue.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])
        self.verbalizer_issue.load_state_dict(state_dict['verbalizer_issue'])        



class JointMLMAdvMlP(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 verbalizer_label:Verbalizer,
                 issue_num:int,
                 #label_num: int,
                 hidden_size:int,
                 pivot_num:int,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.verbalizer_label = verbalizer_label
        self.issue_num = issue_num
        self.keys = list(template.keys())
        self.pivot_num = pivot_num
        self.issue_fc = nn.Linear(hidden_size, issue_num)
        self.pivot_fc = nn.Linear(hidden_size, pivot_num)
        #self.label_emb = nn.Embedding(label_num, hidden_size) #label embedding 可学习的参数，代替token embeddings
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
        self.apply(_init_fc_weights)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha, train=True) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch, out1 = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch),diff=True)
        share_batch, out2 = self.template_share.process_batch(copy.deepcopy(batch),diff=True)
        
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        share_batch['inputs_embeds'] = ReverseLayerF.apply(share_batch['inputs_embeds'], alpha)
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]
        label_outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        issue_outputs = self.plm(**share_batch, output_hidden_states=True, output_attentions = True)[1]  
        issue_logits = self.issue_fc(issue_outputs) 
        if train:
            mlm_outputs = outputs #batch_size * seq_len *hidden_size
            mlm_logits = self.pivot_fc(mlm_outputs)
            mlm_logits = mlm_logits.view(-1, self.pivot_num)

        label_outputs = self.verbalizer_label.gather_outputs(label_outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)
        if train:
            return label_words_logits, issue_logits, mlm_logits, out1, out2
        else:
            return label_words_logits, issue_logits, None, out1, out2
    
    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device
    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):

        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer_label.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        _state_dict['issue_fc'] = self.issue_fc.state_dict()
        _state_dict['pivot_fc'] = self.pivot_fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])
        self.issue_fc.load_state_dict(state_dict['issue_fc'])   
        self.pivot_fc.load_state_dict(state_dict['pivot_fc'])


class JointMLMAdvMlPLabelEmb(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 label_lst: list,
                 issue_num:int,
                 hidden_size:int,
                 pivot_num:int,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.issue_num = issue_num
        self.keys = list(template.keys())
        self.pivot_num = pivot_num
        self.issue_fc = nn.Linear(hidden_size, issue_num)
        self.pivot_fc = nn.Linear(hidden_size, pivot_num)
        self.label_lst = label_lst
        self.label_emb = nn.Embedding(len(self.label_lst), hidden_size) #label embedding 可学习的参数，代替token embeddings
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
        self.apply(_init_fc_weights)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha, train=True) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch,out1 = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch),diff=True)
        share_batch,out2 = self.template_share.process_batch(copy.deepcopy(batch),diff=True)
        
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        share_batch['inputs_embeds'] = ReverseLayerF.apply(share_batch['inputs_embeds'], alpha)

        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]
        label_outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        issue_outputs = self.plm(**share_batch, output_hidden_states=True, output_attentions = True)[1] 
        issue_logits = self.issue_fc(issue_outputs)         
        if train:
            mlm_outputs = outputs #batch_size * seq_len *hidden_size
            mlm_logits = self.pivot_fc(mlm_outputs)
            mlm_logits = mlm_logits.view(-1, self.pivot_num)

        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_tokens = torch.arange(len(self.label_lst)).long().to(self.plm.device)
        batch_size = outputs.size(0)
        label_emb = self.label_emb(label_tokens)
        label_words_logits = torch.mm(outputs_at_mask, label_emb.T)
        label_words_logits = F.sigmoid(label_words_logits)
        if train:
            return label_words_logits, issue_logits, mlm_logits, out1, out2
        else:
            return label_words_logits, issue_logits, None, out1, out2
    
    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device
    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):

        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.__getattr__('template_'+str(k)).state_dict()
        _state_dict['issue_fc'] = self.issue_fc.state_dict()
        _state_dict['pivot_fc'] = self.pivot_fc.state_dict()
        _state_dict['label_emb'] = self.label_emb.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.issue_fc.load_state_dict(state_dict['issue_fc'])
        self.pivot_fc.load_state_dict(state_dict['pivot_fc'])     
        self.label_emb.load_state_dict(state_dict['label_emb']) 



class JointMLMDiff(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 verbalizer_label:Verbalizer,
                 verbalizer_issue:Verbalizer,
                 #label_num: int,
                 hidden_size:int,
                 pivot_num:int,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.verbalizer_label = verbalizer_label
        self.verbalizer_issue = verbalizer_issue
        self.keys = list(template.keys())
        self.pivot_num = pivot_num
        self.fc = nn.Linear(hidden_size, pivot_num)
        #self.label_emb = nn.Embedding(label_num, hidden_size) #label embedding 可学习的参数，代替token embeddings
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
        self.apply(_init_fc_weights)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha, train=True) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch,out1 = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch), diff=True)
        share_batch,out2 = self.template_share.process_batch(copy.deepcopy(batch),diff=True)
        avg_batch = self.template_share.process_batch(copy.deepcopy(batch))

        avg_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        avg_batch = {key: avg_batch[key] for key in avg_batch if key in self.forward_keys}
        issue_batch = {key:issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key:share_batch[key] for key in share_batch if key in self.forward_keys}

        outputs = self.plm(**avg_batch, output_hidden_states=True, output_attentions = True)[0]
        label_outputs = self.template_share.post_processing_outputs(outputs)
        issue_outputs = self.plm(**issue_batch, output_hidden_states=True, output_attentions = True)[0]  #mlm output
        issue_outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(issue_outputs)  
        if train:
            mlm_outputs = outputs #batch_size * seq_len *hidden_size
            mlm_logits = self.fc(mlm_outputs)
            mlm_logits = mlm_logits.view(-1, self.pivot_num)

        label_outputs = self.verbalizer_label.gather_outputs(label_outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)
        issue_outputs = self.verbalizer_issue.gather_outputs(issue_outputs)
        outputs_at_mask = self.extract_at_mask(issue_outputs, batch)
        issue_logits = self.verbalizer_issue.process_logits2(outputs_at_mask, batch=batch)
        if train:
            return label_words_logits, issue_logits, mlm_logits, out1, out2
        else:
            return label_words_logits, issue_logits, None, out1, out2
    
    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device
    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):

        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer_label.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        _state_dict['verbalizer_issue'] = self.verbalizer_issue.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])
        self.verbalizer_issue.load_state_dict(state_dict['verbalizer_issue']) 




# 把issue改成fc, 不用verbalizer
class JointMLMDiffMLP(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 verbalizer_label:Verbalizer,
                 issue_num:int,
                 #label_num: int,
                 hidden_size:int,
                 pivot_num:int,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.verbalizer_label = verbalizer_label
        self.keys = list(template.keys())
        self.pivot_num = pivot_num
        self.issue_fc = nn.Linear(hidden_size, issue_num)
        self.pivot_fc = nn.Linear(hidden_size, pivot_num)
        #self.label_emb = nn.Embedding(label_num, hidden_size) #label embedding 可学习的参数，代替token embeddings
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
        self.apply(_init_fc_weights)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha, train=True) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch,out1 = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch),diff=True)
        share_batch,out2 = self.template_share.process_batch(copy.deepcopy(batch),diff=True)
        avg_batch = self.template_share.process_batch(copy.deepcopy(batch))

        avg_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        avg_batch = {key: avg_batch[key] for key in avg_batch if key in self.forward_keys}
        issue_batch = {key:issue_batch[key] for key in issue_batch if key in self.forward_keys}

        outputs = self.plm(**avg_batch, output_hidden_states=True, output_attentions = True)[0]
        label_outputs = self.template_share.post_processing_outputs(outputs)
        issue_outputs = self.plm(**issue_batch, output_hidden_states=True, output_attentions = True)[1]
        issue_logits = self.issue_fc(issue_outputs) 
        if train:
            mlm_outputs = outputs #batch_size * seq_len *hidden_size
            mlm_logits = self.pivot_fc(mlm_outputs)
            mlm_logits = mlm_logits.view(-1, self.pivot_num)

        label_outputs = self.verbalizer_label.gather_outputs(label_outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)
        if train:
            return label_words_logits, issue_logits, mlm_logits, out1, out2
        else:
            return label_words_logits, issue_logits, None, out1, out2
    
    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device
    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):

        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer_label.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        _state_dict['issue_fc'] = self.issue_fc.state_dict()
        _state_dict['pivot_fc'] = self.pivot_fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])
        self.issue_fc.load_state_dict(state_dict['issue_fc'])
        self.pivot_fc.load_state_dict(state_dict['pivot_fc'])        



# 把label改为可学习参数
class JointMLMDiffMLPLabelEmb(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 label_lst:list,
                 issue_num:int,
                 hidden_size:int,
                 pivot_num:int,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.keys = list(template.keys())
        self.pivot_num = pivot_num
        self.issue_fc = nn.Linear(hidden_size, issue_num)
        self.pivot_fc = nn.Linear(hidden_size, pivot_num)
        self.label_lst = label_lst
        self.label_emb = nn.Embedding(len(self.label_lst), hidden_size) #label embedding 可学习的参数，代替token embeddings
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
        self.apply(_init_fc_weights)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha, train=True) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch,out1 = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch),diff=True)
        share_batch,out2 = self.template_share.process_batch(copy.deepcopy(batch),diff=True)
        avg_batch = self.template_share.process_batch(copy.deepcopy(batch))

        avg_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        avg_batch = {key: avg_batch[key] for key in avg_batch if key in self.forward_keys}
        issue_batch = {key:issue_batch[key] for key in issue_batch if key in self.forward_keys}

        outputs = self.plm(**avg_batch, output_hidden_states=True, output_attentions = True)[0]
        issue_outputs = self.plm(**issue_batch, output_hidden_states=True, output_attentions = True)[1]
        issue_logits = self.issue_fc(issue_outputs) 
        if train:
            mlm_outputs = outputs #batch_size * seq_len *hidden_size
            mlm_logits = self.pivot_fc(mlm_outputs)
            mlm_logits = mlm_logits.view(-1, self.pivot_num)

        label_outputs = outputs
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_tokens = torch.arange(len(self.label_lst)).long().to(self.plm.device)
        batch_size = outputs.size(0)
        #label_tokens = label_tokens.unsqueeze(0).expand(batch_size, -1).to(self.plm.device)
        label_emb = self.label_emb(label_tokens)
        label_words_logits = torch.mm(outputs_at_mask, label_emb.T)
        label_words_logits = F.sigmoid(label_words_logits)        
        if train:
            return label_words_logits, issue_logits, mlm_logits, out1, out2
        else:
            return label_words_logits, issue_logits, None, out1, out2
    
    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device
    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):

        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer_label.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.__getattr__('template_'+str(k)).state_dict()
        _state_dict['issue_fc'] = self.issue_fc.state_dict()
        _state_dict['pivot_fc'] = self.pivot_fc.state_dict()
        _state_dict['label_emb'] = self.label_emb.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.issue_fc.load_state_dict(state_dict['issue_fc'])
        self.pivot_fc.load_state_dict(state_dict['pivot_fc'])     
        self.label_emb.load_state_dict(state_dict['label_emb'])   
