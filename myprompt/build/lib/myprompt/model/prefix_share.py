"""
用p-tuning v2 的方式给每个domain加一个prefix
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


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, 
                pre_seq_len, 
                hidden_size, 
                num_hidden_layers,
                prefix_projection=False,
                prefix_hidden_size=None,
                ):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection: # reparameterazation
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values




class PrefixShareMLM(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 pre_seq_len: Dict[str,int], #prefix_sequence_length
                 verbalizer: Verbalizer,   
                 num_hidden_layers:int,
                 num_attention_heads:int,
                 hidden_size:int,      
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 prefix_projection=False,
                 prefix_hidden_size=None,
                ):
        super().__init__()
        self.plm = plm
        # for k in template:
        #     self.__setattr__('template_'+str(k),template[k])     
        self.pre_seq_len = pre_seq_len
        self.n_layer = num_hidden_layers
        self.n_head = num_attention_heads
        self.n_embd = hidden_size // num_attention_heads
        self.prefix_tokens = {}
        for k in self.pre_seq_len:
            self.prefix_tokens[k] = torch.arange(self.pre_seq_len[k]).long()
            self.__setattr__('prefix_encoder_'+str(k), PrefixEncoder(self.pre_seq_len[k],
                                            hidden_size, self.n_layer))
        self.verbalizer = verbalizer
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.dropout=nn.Dropout(0.1)
        plm_param = 0
        for name, param in self.plm.named_parameters():
            plm_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
            total_param = all_param - plm_param
            print('w/o plm total param is {}'.format(total_param))
        else:
            print('with plm total param is {}'.format(all_param))
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

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


    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        for k in self.pre_seq_len:
            self.__getattr__('prefix_encoder_'+str(k)).train()
        return self

    def get_prompt(self, batch_size,k):
        prefix_tokens = self.prefix_tokens[k].unsqueeze(0).expand(batch_size, -1).to(self.plm.device)
        past_key_values = self.__getattr__('prefix_encoder_'+str(k))(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len[k],
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) #分成一个元组，有num_layers个元素，每一个元素是一层需要用到的past_key_values,之后会作为key,value在计算self-attn时和原来的key,value拼接，详见BertSelfAttention源码
        return past_key_values


    def forward(self, batch: Union[Dict, InputFeatures], k) -> torch.Tensor:
        input_ids,attention_mask = batch['input_ids'], batch['attention_mask']
        batch_size = input_ids.size(0)
        past_key_values_share = self.get_prompt(batch_size=batch_size,k='share')
        prefix_attention_mask_share = torch.ones(batch_size, self.pre_seq_len['share']).to(self.plm.device)
        if k in self.pre_seq_len:
            # 和share过平均
            past_key_values = self.get_prompt(batch_size=batch_size,k=k)
            past_key_values_avg = []
            for i in range(len(past_key_values)):
                past_key_values_avg.append((past_key_values_share[i]+past_key_values[i])/2)
            attention_mask = torch.cat([prefix_attention_mask_share, attention_mask], dim=1)        
        else:# 只用share
            attention_mask = torch.cat([prefix_attention_mask_share, attention_mask],dim=1)
            past_key_values_avg = past_key_values_share
        batch['attention_mask'] = attention_mask
        batch['past_key_values'] = past_key_values_avg

        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]     
        label_outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        
        return label_words_logits

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        _state_dict['plm'] = self.plm.state_dict()
        for k in self.pre_seq_len:
            _state_dict['prefix_encoder_'+str(k)] = self.__getattr__('prefix_encoder_'+str(k)).state_dict()
        _state_dict['verbalizer'] = self.verbalizer.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.pre_seq_len:
            self.__getattr__('prefix_encoder_'+str(k)).load_state_dict(state_dict['prefix_encoder_'+str(k)])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])



def _init_fc_weights(module):
    if isinstance(module,nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)  #Linear层的参数
        if module.bias is not None:
            nn.init.constant_(module.bias.data,0.0) 

# 在encoder之前加grl
class PrefixShareAdvMLM(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 pre_seq_len: Dict[str,int], #prefix_sequence_length
                 verbalizer: Verbalizer, 
                 issue_num: int,  
                 num_hidden_layers:int,
                 num_attention_heads:int,
                 hidden_size:int,      
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 prefix_projection=False,
                 prefix_hidden_size=None,
                ):
        super().__init__()
        self.plm = plm
        # for k in template:
        #     self.__setattr__('template_'+str(k),template[k])     
        self.pre_seq_len = pre_seq_len
        self.n_layer = num_hidden_layers
        self.n_head = num_attention_heads
        self.n_embd = hidden_size // num_attention_heads
        self.prefix_tokens = {}
        for k in self.pre_seq_len:
            self.prefix_tokens[k] = torch.arange(self.pre_seq_len[k]).long()
            self.__setattr__('prefix_encoder_'+str(k), PrefixEncoder(self.pre_seq_len[k],
                                            hidden_size, self.n_layer))
        self.verbalizer= verbalizer
        self.fc = nn.Linear(hidden_size, issue_num)
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.dropout=nn.Dropout(0.1)
        plm_param = 0
        for name, param in self.plm.named_parameters():
            plm_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
            total_param = all_param - plm_param
            print('w/o plm total param is {}'.format(total_param))
        else:
            print('with plm total param is {}'.format(all_param))
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
        self.apply(_init_fc_weights)

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


    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        for k in self.pre_seq_len:
            self.__getattr__('prefix_encoder_'+str(k)).train()
        return self

    def get_prompt(self, batch_size,k):
        prefix_tokens = self.prefix_tokens[k].unsqueeze(0).expand(batch_size, -1).to(self.plm.device)
        past_key_values = self.__getattr__('prefix_encoder_'+str(k))(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len[k],
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) #分成一个元组，有num_layers个元素，每一个元素是一层需要用到的past_key_values,之后会作为key,value在计算self-attn时和原来的key,value拼接，详见BertSelfAttention源码
        return past_key_values


    def forward(self, batch: Union[Dict, InputFeatures], k, alpha) -> torch.Tensor:
        adv_batch = copy.deepcopy(batch)
        input_ids,attention_mask = batch['input_ids'], batch['attention_mask']
        batch_size = input_ids.size(0)
        past_key_values_share = self.get_prompt(batch_size=batch_size,k='share')
        prefix_attention_mask_share = torch.ones(batch_size, self.pre_seq_len['share']).to(self.plm.device)
        if k in self.pre_seq_len:
            # 和share过平均
            past_key_values = self.get_prompt(batch_size=batch_size,k=k)
            past_key_values_avg = []
            for i in range(len(past_key_values)):
                past_key_values_avg.append((past_key_values_share[i]+past_key_values[i])/2)
            attention_mask_share = torch.cat([prefix_attention_mask_share, attention_mask], dim=1)        
        else:# 只用share
            attention_mask_share = torch.cat([prefix_attention_mask_share, attention_mask],dim=1)
            past_key_values_avg = past_key_values_share
        batch['attention_mask'] = attention_mask_share
        batch['past_key_values'] = past_key_values_avg

        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        ## adv, 只用share prefix
        adv_past_key_values = []
        for i in range(len(past_key_values_share)):
            adv_past_key_values.append(ReverseLayerF.apply(past_key_values_share[i], alpha))
        adv_batch['attention_mask'] = attention_mask_share
        adv_batch['past_key_values'] = adv_past_key_values
        adv_input_batch={key: adv_batch[key] for key in adv_batch if key in self.forward_keys}

        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]     
        label_outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)

        adv_outputs = self.plm(**adv_input_batch, output_hidden_states=True, output_attentions = True)[1]
        adv_logits = self.fc(adv_outputs)
        return label_words_logits, adv_logits

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer_label.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        _state_dict['plm'] = self.plm.state_dict()
        for k in self.pre_seq_len:
            _state_dict['prefix_encoder_'+str(k)] = self.__getattr__('prefix_encoder_'+str(k)).state_dict()
        _state_dict['verbalizer'] = self.verbalizer.state_dict()
        _state_dict['fc'] = self.fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.pre_seq_len:
            self.__getattr__('prefix_encoder_'+str(k)).load_state_dict(state_dict['prefix_encoder_'+str(k)])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])
        self.fc.load_state_dict(state_dict['fc'])


# 在encoder之后、fc之前加grl
class PrefixShareAdvMLM2(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel, 
                 pre_seq_len: Dict[str,int], #prefix_sequence_length
                 verbalizer: Verbalizer, 
                 issue_num:int,  
                 num_hidden_layers:int,
                 num_attention_heads:int,
                 hidden_size:int,      
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 prefix_projection=False,
                 prefix_hidden_size=None,
                ):
        super().__init__()
        self.plm = plm
        # for k in template:
        #     self.__setattr__('template_'+str(k),template[k])     
        self.pre_seq_len = pre_seq_len
        self.n_layer = num_hidden_layers
        self.n_head = num_attention_heads
        self.n_embd = hidden_size // num_attention_heads
        self.prefix_tokens = {}
        for k in self.pre_seq_len:
            self.prefix_tokens[k] = torch.arange(self.pre_seq_len[k]).long()
            self.__setattr__('prefix_encoder_'+str(k), PrefixEncoder(self.pre_seq_len[k],
                                            hidden_size, self.n_layer))
        self.verbalizer = verbalizer
        self.fc = nn.Linear(hidden_size, issue_num)
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.dropout=nn.Dropout(0.1)
        plm_param = 0
        for name, param in self.plm.named_parameters():
            plm_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
            total_param = all_param - plm_param
            print('w/o plm total param is {}'.format(total_param))
        else:
            print('with plm total param is {}'.format(all_param))
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args
        self.apply(_init_fc_weights)

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


    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        for k in self.pre_seq_len:
            self.__getattr__('prefix_encoder_'+str(k)).train()
        return self

    def get_prompt(self, batch_size,k):
        prefix_tokens = self.prefix_tokens[k].unsqueeze(0).expand(batch_size, -1).to(self.plm.device)
        past_key_values = self.__getattr__('prefix_encoder_'+str(k))(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len[k],
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) #分成一个元组，有num_layers个元素，每一个元素是一层需要用到的past_key_values,之后会作为key,value在计算self-attn时和原来的key,value拼接，详见BertSelfAttention源码
        return past_key_values


    def forward(self, batch: Union[Dict, InputFeatures], k, alpha) -> torch.Tensor:
        adv_batch = copy.deepcopy(batch)
        input_ids,attention_mask = batch['input_ids'], batch['attention_mask']
        batch_size = input_ids.size(0)
        past_key_values_share = self.get_prompt(batch_size=batch_size,k='share')
        prefix_attention_mask_share = torch.ones(batch_size, self.pre_seq_len['share']).to(self.plm.device)
        if k in self.pre_seq_len:
            # 和share过平均
            past_key_values = self.get_prompt(batch_size=batch_size,k=k)
            past_key_values_avg = []
            for i in range(len(past_key_values)):
                past_key_values_avg.append((past_key_values_share[i]+past_key_values[i])/2)
            attention_mask_avg = torch.cat([prefix_attention_mask_share, attention_mask], dim=1)        
        else:# 只用share
            attention_mask_avg = torch.cat([prefix_attention_mask_share, attention_mask],dim=1)
            past_key_values_avg = past_key_values_share
        batch['attention_mask'] = attention_mask_avg
        batch['past_key_values'] = past_key_values_avg
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}

        adv_batch['attention_mask'] = attention_mask_avg
        adv_batch['past_key_values'] = past_key_values_share
        adv_input_batch = {key: adv_batch[key] for key in adv_batch if key in self.forward_keys}

        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]     
        label_outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)

        adv_outputs = self.plm(**adv_input_batch, output_hidden_states=True, output_attentions = True)[1]
        adv_logits = self.fc(adv_outputs)

        return label_words_logits, adv_logits


    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        _state_dict['plm'] = self.plm.state_dict()
        for k in self.pre_seq_len:
            _state_dict['prefix_encoder_'+str(k)] = self.__getattr__('prefix_encoder_'+str(k)).state_dict()
        _state_dict['verbalizer'] = self.verbalizer.state_dict()
        _state_dict['fc'] = self.fc.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.pre_seq_len:
            self.__getattr__('prefix_encoder_'+str(k)).load_state_dict(state_dict['prefix_encoder_'+str(k)])
        self.verbalizer.load_state_dict(state_dict['verbalizer'])
        self.fc.load_state_dict(state_dict['fc'])


        

# 用verbalizer处理issue
class PrefixShareAdvMLM3(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel, 
                 pre_seq_len: Dict[str,int], #prefix_sequence_length
                 verbalizer_label: Verbalizer, 
                 verbalizer_issue:Verbalizer,
                 num_hidden_layers:int,
                 num_attention_heads:int,
                 hidden_size:int,      
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                 prefix_projection=False,
                 prefix_hidden_size=None,
                ):
        super().__init__()
        self.plm = plm
        # for k in template:
        #     self.__setattr__('template_'+str(k),template[k])     
        self.pre_seq_len = pre_seq_len
        self.n_layer = num_hidden_layers
        self.n_head = num_attention_heads
        self.n_embd = hidden_size // num_attention_heads
        self.prefix_tokens = {}
        for k in self.pre_seq_len:
            self.prefix_tokens[k] = torch.arange(self.pre_seq_len[k]).long()
            self.__setattr__('prefix_encoder_'+str(k), PrefixEncoder(self.pre_seq_len[k],
                                            hidden_size, self.n_layer))
        self.verbalizer_label= verbalizer_label
        self.verbalizer_issue = verbalizer_issue
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        self.dropout=nn.Dropout(0.1)
        plm_param = 0
        for name, param in self.plm.named_parameters():
            plm_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
            total_param = all_param - plm_param
            print('w/o plm total param is {}'.format(total_param))
        else:
            print('with plm total param is {}'.format(all_param))
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

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


    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        for k in self.pre_seq_len:
            self.__getattr__('prefix_encoder_'+str(k)).train()
        return self

    def get_prompt(self, batch_size,k):
        prefix_tokens = self.prefix_tokens[k].unsqueeze(0).expand(batch_size, -1).to(self.plm.device)
        past_key_values = self.__getattr__('prefix_encoder_'+str(k))(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len[k],
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) #分成一个元组，有num_layers个元素，每一个元素是一层需要用到的past_key_values,之后会作为key,value在计算self-attn时和原来的key,value拼接，详见BertSelfAttention源码
        return past_key_values


    def forward(self, batch: Union[Dict, InputFeatures], k, alpha) -> torch.Tensor:
        adv_batch = copy.deepcopy(batch)
        input_ids,attention_mask = batch['input_ids'], batch['attention_mask']
        batch_size = input_ids.size(0)
        past_key_values_share = self.get_prompt(batch_size=batch_size,k='share')
        prefix_attention_mask_share = torch.ones(batch_size, self.pre_seq_len['share']).to(self.plm.device)
        if k in self.pre_seq_len:
            # 和share过平均
            past_key_values = self.get_prompt(batch_size=batch_size,k=k)
            past_key_values_avg = []
            for i in range(len(past_key_values)):
                past_key_values_avg.append((past_key_values_share[i]+past_key_values[i])/2)
            attention_mask_share = torch.cat([prefix_attention_mask_share, attention_mask], dim=1)        
        else:# 只用share
            attention_mask_share = torch.cat([prefix_attention_mask_share, attention_mask],dim=1)
            past_key_values_avg = past_key_values_share
        batch['attention_mask'] = attention_mask_share
        batch['past_key_values'] = past_key_values_avg

        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        ## adv, 只用share prefix
        adv_past_key_values = []
        for i in range(len(past_key_values_share)):
            adv_past_key_values.append(ReverseLayerF.apply(past_key_values_share[i], alpha))
        adv_batch['attention_mask'] = attention_mask_share
        adv_batch['past_key_values'] = adv_past_key_values
        adv_input_batch={key: adv_batch[key] for key in adv_batch if key in self.forward_keys}

        outputs = self.plm(**input_batch, output_hidden_states=True, output_attentions = True)[0]     
        label_outputs = self.verbalizer_label.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(label_outputs, batch)
        label_words_logits = self.verbalizer_label.process_outputs(outputs_at_mask, batch=batch)

        adv_outputs = self.plm(**adv_input_batch, output_hidden_states=True, output_attentions = True)[0]
        adv_label_outputs = self.verbalizer_issue.gather_outputs(adv_outputs)
        outputs_at_mask = self.extract_at_mask(adv_label_outputs, adv_batch)
        adv_logits = self.verbalizer_issue.process_outputs(outputs_at_mask, batch=adv_batch)

        return label_words_logits, adv_logits

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer_label.tokenizer

    def state_dict(self):
        """ Save the model using template, plm and verbalizer's save methods."""
        _state_dict = {}
        _state_dict['plm'] = self.plm.state_dict()
        for k in self.pre_seq_len:
            _state_dict['prefix_encoder_'+str(k)] = self.__getattr__('prefix_encoder_'+str(k)).state_dict()
        _state_dict['verbalizer_label'] = self.verbalizer_label.state_dict()
        _state_dict['verbalizer_issue'] = self.verbalizer_issue.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template, plm and verbalizer's load methods."""
        if 'plm' in state_dict and not self.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        for k in self.pre_seq_len:
            self.__getattr__('prefix_encoder_'+str(k)).load_state_dict(state_dict['prefix_encoder_'+str(k)])
        self.verbalizer_label.load_state_dict(state_dict['verbalizer_label'])
        self.verbalizer_issue.load_state_dict(state_dict['verbalizer_issue'])
