"""
label embedding 匹配
用MLM rep 而不是过了30522后的词概率分布
"""

import torch
import torch.nn as nn
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

class PromptMLMModel(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        self.template = template
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions=True)[0]
        outputs = self.template.post_processing_outputs(outputs)
        return outputs

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures]) -> Dict:
        r"""Will be used in generation
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch


def _init_fc_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)  #Linear层的参数
        if module.bias is not None:
            nn.init.constant_(module.bias.data,0.0) 
# 0. mlm后直接过num_classes 的全连接层, 不要label embedding
class PromptwoVerbalizer(nn.Module):
    def __init__(self,
                plm: PreTrainedModel,
                template: Template,
                num_classes: int,
                freeze_plm: bool=False,
                plm_eval_mode: bool=False,
                hidden_size: int =768
                ):
        super().__init__()
        self.prompt_model = PromptMLMModel(plm, template, freeze_plm, plm_eval_mode)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.apply(_init_fc_weights)

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

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

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" 
        """
        outputs = self.prompt_model(batch)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_logits = self.fc(outputs_at_mask)
        return label_logits                

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        _state_dict['template'] = self.template.state_dict()
        _state_dict['fc'] = self.fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        self.template.load_state_dict(state_dict['template'])
        self.fc.load_state_dict(state_dict['fc'])

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.fc.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.fc.cpu()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")                            




### 1. 用正常的模板
class PromptForLabelMatching(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Template,
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptMLMModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

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

    def forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" 
        """
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits                

    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        _state_dict['template'] = self.template.state_dict()
        _state_dict['verbalizer'] = self.verbalizer.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        self.template.load_state_dict(state_dict['template'])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.verbalizer.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.verbalizer.cpu()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")                            



### 2.用share的模板

### 2.1 普通share，不加GRL
class PromptShareModel(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue) -> torch.Tensor:

        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        share_batch = self.template_share.process_batch(copy.deepcopy(batch))
        out1, out2 = issue_batch['inputs_embeds'], share_batch['inputs_embeds']
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        return outputs


class PromptForShare(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer_label: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptShareModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer_label = verbalizer_label
        self.keys=list(template.keys()) 

    @property
    def plm(self):
        return self.prompt_model.plm
    
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

    def forward(self, batch: Union[Dict, InputFeatures], issue) -> torch.Tensor:
        r""" 
        """
        outputs = self.prompt_model(batch, issue)
        label_outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)        
        return label_words_logits

    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures], issue) -> torch.Tensor:
        outputs, issue_outputs = self.prompt_model(batch, issue)
        outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.prompt_model.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.prompt_model.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])




### 2.2 加上GRL
class PromptSharedMLMModel(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        share_batch = self.template_share.process_batch(copy.deepcopy(batch))
        out1, out2 = issue_batch['inputs_embeds'], share_batch['inputs_embeds']
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        share_batch['inputs_embeds'] = ReverseLayerF.apply(share_batch['inputs_embeds'], alpha)
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        outputs2 = self.plm(**share_batch, output_hidden_states=True, output_attentions = True)[0]  #mlm output
        outputs2 = self.template_share.post_processing_outputs(outputs2)        
        return outputs, outputs2, out1, out2

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures],issue) -> Dict:
        r"""Will be used in generation
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        share_batch = self.template_share.process_batch(batch)
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        return input_batch, share_batch




class PromptForSharedMLMTemplate(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer_label: Verbalizer,
                 verbalizer_issue: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptSharedMLMModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer_label = verbalizer_label
        self.verbalizer_issue = verbalizer_issue
        self.keys=list(template.keys()) 

    @property
    def plm(self):
        return self.prompt_model.plm
    
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

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha) -> torch.Tensor:
        r""" 
        """
        outputs, issue_outputs, out1, out2 = self.prompt_model(batch, issue, alpha)
        label_outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)

    
        issue_outputs = self.verbalizer_issue.gather_outputs(issue_outputs)
        outputs_at_mask = self.extract_at_mask(issue_outputs, batch)
        #issue_logits = self.verbalizer_issue.process_outputs(outputs_at_mask, batch=batch)
        issue_logits = self.verbalizer_issue.process_logits2(outputs_at_mask, batch=batch)
        
        return label_words_logits, issue_logits, out1, out2

    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures], issue, alpha) -> torch.Tensor:
        outputs, issue_outputs = self.prompt_model(batch, issue, alpha)
        outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask


    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.prompt_model.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        _state_dict['verbalizer_issue'] = self.verbalizer_issue.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.prompt_model.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])
        self.verbalizer_issue.load_state_dict(state_dict['verbalizer_issue'])


## issue label不用 verbalizer, 直接用mlp 处理logits
def _init_fc_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)  #Linear层的参数
        if module.bias is not None:
            nn.init.constant_(module.bias.data,0.0) 


class PromptForSharedMLMTemplateMLP(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer_label: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 hidden_dim: int = 768,
                 issue_class: int = None
                ):
        super().__init__()
        self.prompt_model = PromptSharedMLMModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer_label = verbalizer_label
        self.issue_fc = nn.Linear(hidden_dim,issue_class)
        self.keys=list(template.keys()) 
        self.apply(_init_fc_weights)

    @property
    def plm(self):
        return self.prompt_model.plm
    
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

    def forward(self, batch: Union[Dict, InputFeatures], issue, alpha) -> torch.Tensor:
        r""" 
        """
        outputs, issue_outputs, out1, out2 = self.prompt_model(batch, issue, alpha)
        label_outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)

        outputs_at_mask = self.extract_at_mask(issue_outputs, batch)
        issue_logits = self.issue_fc(outputs_at_mask)
        
        return label_words_logits, issue_logits, out1, out2

    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures], issue, alpha) -> torch.Tensor:
        outputs, issue_outputs = self.prompt_model(batch, issue, alpha)
        outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask


    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.prompt_model.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        _state_dict['issue_fc'] = self.issue_fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.prompt_model.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])
        self.issue_fc.load_state_dict(state_dict['issue_fc'])


## PromptForNPNet

class PromptIssueMLMModel(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: Union[Dict, InputFeatures], issue) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[1]  #用pooled output, 不用sequence output
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)      
        return outputs

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures],issue) -> Dict:
        r"""Will be used in generation
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        return input_batch



class PromptForNPNet(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 hidden_dim: int = 768,
                 frame_class: int = None
                ):
        super().__init__()
        self.prompt_model = PromptIssueMLMModel(plm, template, freeze_plm, plm_eval_mode)
        self.fc = nn.Linear(hidden_dim, frame_class)
        self.keys=list(template.keys()) 
        self.apply(_init_fc_weights)

    @property
    def plm(self):
        return self.prompt_model.plm
    
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

    def forward(self, batch: Union[Dict, InputFeatures], issue) -> torch.Tensor:
        r""" 
        """
        outputs= self.prompt_model(batch, issue)
        label_outputs = outputs
        #outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        #label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)
        label_words_logits = self.fc(label_outputs)
        label_words_logits=torch.sigmoid(label_words_logits)
        
        return label_words_logits

    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures], issue, alpha) -> torch.Tensor:
        outputs = self.prompt_model(batch, issue, alpha)
        outputs = outputs
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.prompt_model.__getattr__('template_'+str(k)).state_dict()
        _state_dict['fc'] = self.fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.prompt_model.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.fc.load_state_dict(state_dict['fc'])



#frame prediction和pivot prediction loss合并
class PromptJointIssueMLMModel(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch1: Union[Dict, InputFeatures], batch2, issue) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch1))
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        masked_out = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]

        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch2))
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}        
        pooled_out = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[1]
        return masked_out, pooled_out

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures],issue) -> Dict:
        r"""Will be used in generation
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        return input_batch



class PromptForJointNPNet(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer : Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 hidden_dim: int = 768,
                 frame_class: int = None
                ):
        super().__init__()
        self.prompt_model = PromptJointIssueMLMModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer
        self.fc = nn.Linear(hidden_dim, frame_class)
        self.keys=list(template.keys()) 
        self.apply(_init_fc_weights)

    @property
    def plm(self):
        return self.prompt_model.plm
    
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

    def forward(self, batch1: Union[Dict, InputFeatures], batch2, issue) -> torch.Tensor:
        r""" 
        """
        masked_out, pooled_out = self.prompt_model(batch1, batch2, issue)
        label_outputs = masked_out
        outputs_at_mask = self.extract_at_mask(label_outputs, batch1)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch1)

        pooled_out = self.fc(pooled_out)
        pooled_out = torch.sigmoid(pooled_out)
        return label_words_logits, pooled_out

    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures], issue, alpha) -> torch.Tensor:
        outputs = self.prompt_model(batch, issue, alpha)
        outputs = outputs
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.prompt_model.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer'] = self.verbalizer.state_dict()
        _state_dict['fc'] = self.fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.prompt_model.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])
        self.fc.load_state_dict(state_dict['fc'])


## transfer+ NP net联合训练
class PromptJointModel(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,PtuningTemplate],
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        for k in template:
            self.__setattr__('template_'+str(k),template[k])           
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch1: Union[Dict, InputFeatures], batch2, issue, alpha) -> torch.Tensor:
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch1))
        share_batch = self.template_share.process_batch(copy.deepcopy(batch1))
        out1, out2 = issue_batch['inputs_embeds'], share_batch['inputs_embeds']
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        share_batch['inputs_embeds'] = ReverseLayerF.apply(share_batch['inputs_embeds'], alpha)
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        outputs2 = self.plm(**share_batch, output_hidden_states=True, output_attentions = True)[0]  #mlm output
        outputs2 = self.template_share.post_processing_outputs(outputs2)      

        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch2))
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}        
        pooled_out = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[1]        

        return outputs, outputs2, pooled_out, out1, out2

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures],issue) -> Dict:
        r"""Will be used in generation
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        share_batch = self.template_share.process_batch(batch)
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        return input_batch, share_batch


class PromptForJointModel(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer_label: Verbalizer,
                 verbalizer_issue: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 hidden_dim:int=768,
                 frame_class: int =None
                ):
        super().__init__()
        self.prompt_model = PromptJointModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer_label = verbalizer_label
        self.verbalizer_issue = verbalizer_issue
        self.fc = nn.Linear(hidden_dim, frame_class)
        self.keys=list(template.keys()) 
        self.apply(_init_fc_weights)

    @property
    def plm(self):
        return self.prompt_model.plm
    
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

    def forward(self, batch1: Union[Dict, InputFeatures],  batch2, issue, alpha) -> torch.Tensor:
        r""" 
        """
        outputs, issue_outputs, pooled_out, out1, out2 = self.prompt_model(batch1, batch2, issue, alpha)
        label_outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch1)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch1)

    
        issue_outputs = self.verbalizer_issue.gather_outputs(issue_outputs)
        outputs_at_mask = self.extract_at_mask(issue_outputs, batch1)
        issue_logits = self.verbalizer_issue.process_outputs(outputs_at_mask, batch=batch1)

        pooled_out=self.fc(pooled_out)
        pooled_out= torch.sigmoid(pooled_out)
        return label_words_logits, issue_logits,pooled_out, out1, out2

    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures], issue, alpha) -> torch.Tensor:
        outputs, issue_outputs = self.prompt_model(batch, issue, alpha)
        outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.prompt_model.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        _state_dict['verbalizer_issue'] = self.verbalizer_issue.state_dict()
        _state_dict['fc'] = self.fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.prompt_model.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])
        self.verbalizer_issue.load_state_dict(state_dict['verbalizer_issue'])
        self.fc.load_state_dict(state_dict['fc'])