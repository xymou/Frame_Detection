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
from myprompt.template import Template
from myprompt.verbalizer import Verbalizer
from myprompt.plm.utils import TokenizerWrapper
from collections import defaultdict
from myprompt.utils.logging import logger
from myprompt.utils.utils import round_list, signature
from transformers import  AdamW, get_linear_schedule_with_warmup
from myprompt.model.base import PromptModel
from myprompt.model.classification import PromptForClassification

def _init_fc_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)  #Linear层的参数
        if module.bias is not None:
            nn.init.constant_(module.bias.data,0.0) 

#1.用GF的label words，直接加一个线性层硬调
class PromptForTransferClassification(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Template,
                 verbalizer: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 in_classes: int =15,
                 out_classes: int =9                 
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer
        self.fc = nn.Linear(in_classes, out_classes)
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
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        logits = self.fc(label_words_logits)
        return logits                

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
        _state_dict['fc'] = self.fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        self.template.load_state_dict(state_dict['template'])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])
        self.fc.load_state_dict(state_dict['fc'])

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template.cuda()
            self.verbalizer.cuda()
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
            self.verbalizer.cpu()
            self.fc.cpu()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")           


#2.用GF的PLM参数，用DF自己的label word(直接从vocab上而不是15个class上开始处理)
# 见verbalizer.transfer_verbalizer
        