from myprompt.model.classification import PromptForClassification
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
from myprompt.template import Template, PtuningTemplate
from myprompt.verbalizer import Verbalizer
from myprompt.plm.utils import TokenizerWrapper
from collections import defaultdict
from myprompt.utils.logging import logger
from myprompt.utils.utils import round_list, signature
from transformers import  AdamW, get_linear_schedule_with_warmup
from myprompt.model.base import PromptModel
import copy
from myprompt.utils.utils import ReverseLayerF
import torch.nn.functional as F


class PromptSharedModel(nn.Module):

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
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        share_batch = self.template_share.process_batch(batch)
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        return outputs

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures],issue) -> Dict:
        r"""Will be used in generation
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(batch)
        share_batch = self.template_share.process_batch(batch)
        batch = {key:torch.mean(torch.stack([issue_batch[key],share_batch[key]],dim=0), dim=0) for key in self.forward_keys}
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch

                

########### 用多个template的
class PromptForSharedTemplate(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptSharedModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer  
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
        
        for k in self.keys:
            _state_dict['template_'+str(k)] = self.prompt_model.__getattr__('template_'+str(k)).state_dict()
        _state_dict['verbalizer'] = self.verbalizer.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.keys:
            self.prompt_model.__getattr__('template_'+str(k)).load_state_dict(state_dict['template_'+str(k)])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])

                        


################ 单纯加上了issue预测目标的, 训练时loss1-alpha*loss2
class PromptMergedModel(nn.Module):

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
        r""" 
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position. 
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        share_batch = self.template_share.process_batch(batch)
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        outputs2 = self.plm(**share_batch, output_hidden_states=True)
        outputs2 = self.template_share.post_processing_outputs(outputs2)
        return outputs, outputs2

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures],issue) -> Dict:
        r"""Will be used in generation
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        share_batch = self.template_share.process_batch(batch)
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        return input_batch, share_batch



class PromptForMergedGoal(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer_label: Verbalizer,
                 verbalizer_issue: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptMergedModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer_label = verbalizer_label
        self.verbalizer_issue = verbalizer_issue
        self.keys = list(template.keys())

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
        outputs, issue_outputs = self.prompt_model(batch, issue)
        label_outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)

    
        issue_outputs = self.verbalizer_issue.gather_outputs(issue_outputs)
        outputs_at_mask = self.extract_at_mask(issue_outputs, batch)
        issue_logits = self.verbalizer_issue.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits, issue_logits

    def predict(self):
        pass
    
    def forward_without_verbalize(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        outputs, issue_outputs = self.prompt_model(batch)
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

                      


##########加上了梯度反转层的 DANN, 训练时候loss1 + loss2
class PromptDANNModel(nn.Module):

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
        share_batch = self.template_share.process_batch(batch)
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        share_batch['inputs_embeds'] = ReverseLayerF.apply(share_batch['inputs_embeds'], alpha)
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        outputs2 = self.plm(**share_batch, output_hidden_states=True)
        outputs2 = self.template_share.post_processing_outputs(outputs2)
        return outputs, outputs2

    def prepare_model_inputs(self, batch: Union[Dict, InputFeatures],issue) -> Dict:
        r"""Will be used in generation
        """
        issue_batch = self.__getattr__('template_'+str(issue)).process_batch(copy.deepcopy(batch))
        share_batch = self.template_share.process_batch(batch)
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        return input_batch, share_batch

class PromptForDANN(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer_label: Verbalizer,
                 verbalizer_issue: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptDANNModel(plm, template, freeze_plm, plm_eval_mode)
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
        outputs, issue_outputs = self.prompt_model(batch, issue, alpha)
        label_outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)
        #label_words_logits = torch.sigmoid(label_words_logits)

    
        issue_outputs = self.verbalizer_issue.gather_outputs(issue_outputs)
        outputs_at_mask = self.extract_at_mask(issue_outputs, batch)
        issue_logits = self.verbalizer_issue.process_outputs(outputs_at_mask, batch=batch)
        #issue_logits = F.softmax(issue_logits, dim =0)
        return label_words_logits, issue_logits

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




##########加上了公共和私有prompt之间的diff loss,需要输出生成的提示
class PromptDiffDANNModel(nn.Module):

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
        share_batch = self.template_share.process_batch(batch)
        out1, out2 = issue_batch['inputs_embeds'], share_batch['inputs_embeds']
        issue_batch['inputs_embeds'] = (issue_batch['inputs_embeds']+ share_batch['inputs_embeds'])/2
        input_batch = {key: issue_batch[key] for key in issue_batch if key in self.forward_keys}
        share_batch = {key: share_batch[key] for key in share_batch if key in self.forward_keys}
        share_batch['inputs_embeds'] = ReverseLayerF.apply(share_batch['inputs_embeds'], alpha)
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions=True)
        outputs = self.__getattr__('template_'+str(issue)).post_processing_outputs(outputs)
        outputs2 = self.plm(**share_batch, output_hidden_states=True, output_attentions=True)
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

class PromptForDiffDANN(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Dict[str,Template], #issue:template
                 verbalizer_label: Verbalizer,
                 verbalizer_issue: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptDiffDANNModel(plm, template, freeze_plm, plm_eval_mode)
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
        #label_words_logits = torch.sigmoid(label_words_logits)

    
        issue_outputs = self.verbalizer_issue.gather_outputs(issue_outputs)
        outputs_at_mask = self.extract_at_mask(issue_outputs, batch)
        issue_logits = self.verbalizer_issue.process_outputs(outputs_at_mask, batch=batch)
        #issue_logits = F.softmax(issue_logits, dim =0)
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


