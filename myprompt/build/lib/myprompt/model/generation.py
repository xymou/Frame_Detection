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
from transformers.generation_utils import GenerationMixin
from yacs.config import CfgNode

class PromptForGeneration(nn.Module, GenerationMixin):
    r'''``PromptModel`` with generation loss caculation and generation utils integrated.
    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for generation, e.g. GPT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``PrefixTemplate``.
        tokenizer (:obj:`Tokenizer`): A ``Tokenizer`` of the current model.
        gen_config (:obj:`CfgNode`): The generation configs to pass into `GenerationMixin.generate <https://huggingface.co/transformers/_modules/transformers/generation_utils.html#GenerationMixin.generate>`_
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train. 
    '''
    def __init__(self,
                 plm: PreTrainedModel, 
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool = False,
                 gen_config: Optional[CfgNode] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                ):
        super().__init__()
        self.freeze_plm = freeze_plm
        if tokenizer is None:
            assert template.tokenizer is not None, "Tokenizer can't be set from input args or template"
            self.tokenizer = template.tokenizer
        else:
            self.tokenizer = tokenizer
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.config = plm.config
        if gen_config:
            for key in gen_config:
                setattr(self.config, key, gen_config[key])
        self.in_generation_function = False

    @property
    def plm(self):
        return self.prompt_model.plm
    
    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self):
        return self.plm.device

    def shift_logits_and_labels(self, 
                                logits, 
                                loss_ids, 
                                reference_ids):

        r"""
        Left shift the label, and make label of the positions that are
        not loss position to -100, which is the ignore index in pytorch's
        loss function.
        Args:
            logits (:obj:`torch.Tensor`):
            batch (:obj:InputFeatures): The input features of batchified data sequences.
        
        Returns:
            shift_logits (:obj:`torch.Tensor`):
            shift_input_ids (:obj:`List[int]`):
        """
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_loss_ids = loss_ids[..., 1:].contiguous()
        shift_input_ids = reference_ids[..., 1:].contiguous()
        shift_input_ids = torch.where(shift_loss_ids>0, shift_input_ids, -100)
        return shift_logits, shift_input_ids

    def forward(self, *args, **kwargs):
        r"""In generation process, it will use the plm's forward function.
        This is because, in the first step we will directly call the process_batch function to 
        generate initial input with the template, after that the all template
        have been processed into the past_key_value,
        then we can use the normal generation function. 
        In learning process, the forward is linked to ``_forward`` functions.
        in which the loss will be calculated for all the positions in the same time. 
        """
        if self.in_generation_function:
            return self.plm.forward(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def _forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
        r""" 
        This is the forward method of the training of generation in prompt-learning framework. 
        
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        
        Returns:
            loss(:obj:torch.Tensor): The loss of the current generation procedure.
        """
        if self.config.is_encoder_decoder:
            reference_ids = batch['decoder_input_ids']
        else:
            reference_ids = batch['input_ids']  # in case in some template, these field is dropped
        outputs = self.prompt_model(batch)
        logits = outputs.logits
        logits, labels = self.shift_logits_and_labels(logits, batch['loss_ids'], reference_ids)
        batch_size, seq_len, vocab_size = logits.shape
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss = loss.view(batch_size, -1).sum(dim=-1) #TODO support more objectives
        loss = loss.mean()
        return loss
    
    
    def generate(self, batch: Union[Dict, InputFeatures], **generation_kwargs):
        r""" This function wraps the generate() methods in parent class ``GenerationMixin``.
        Forward uses the ``PretrainedModel``'s forward method. 
        generation_kwargs include all the parameters that are passed in to 
        ``transformers.generation_util.GenerationMixin.generate``
    
        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        
        Returns:
            output_sequences (:obj:List[torch.Tensor]): The raw sequences generated by the generation model.
            generated_sentences (:obj:List[torch.Tensor]): The generated sentences that have been post-processed.
        """
        input_generation_kwargs = {key: value for key,value in generation_kwargs.items() if key in signature(GenerationMixin.generate).args}
        if self.config.is_encoder_decoder:
            loss_ids_start = batch['loss_ids'].argmax(dim=-1)
            assert loss_ids_start.min() == loss_ids_start.max(), "The generation start from different position in a batch."
            batch['decoder_input_ids'] = batch['decoder_input_ids'][:, :loss_ids_start.min()+1]
            input_length = batch['decoder_input_ids'].size(1)
            batch_size = batch['decoder_input_ids'].size(0)

            self.generate_ith_token = 0
            self.in_generation_function = True
            output_sequences = super().generate(**batch, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            self.in_generation_function = False
            output_sequences = output_sequences.cpu().tolist()
            generated_sentences = self.post_processing(output_sequences=output_sequences, input_lengths=input_length)
        else:
            input_length = batch['input_ids'].size(1)
            batch_size = batch['input_ids'].size(0)
            
            # Currently huggingface transformers only support single sample generation, or padding to the left (instead of the right).
            # because it will only extract the last position of the output 
            # generate one_by_one
            if 'input_ids_len' in batch:
                input_real_lens = batch['input_ids_len']
            else:
                input_real_lens = torch.sum((batch['input_ids'] != self.tokenizer.pad_token_id).to(torch.int), dim=-1)
            output_sequences = []
            for instance_id in range(batch_size):  
                # remove the pad token 
                instance = {key: batch[key][instance_id:instance_id+1][:,:input_real_lens[instance_id]] for key in batch if isinstance(batch[key], torch.Tensor) and batch[key].shape[:2]==torch.Size([batch_size, input_length])}
                self.generate_ith_token = 0
                self.in_generation_function = True
                output_sequence = super().generate(**instance, **input_generation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
                self.in_generation_function = False
                output_sequences.extend(output_sequence.cpu().tolist()) # TODO: to support generate multiple sentence
            generated_sentences = self.post_processing(output_sequences=output_sequences, input_lengths=input_real_lens.cpu().tolist())
        return output_sequences, generated_sentences
    


    def post_processing(self, output_sequences, input_lengths):
        r"""
            Post-process the sequences generated by the generation model.
            Args:
                output_sequences (:obj:`torch.Tensor`): The raw sequences generated by the generation model.
                input_lengths (:obj:`int` or `list`): The length(s) of the input sequence.
            
            Returns:
                :obj:`List`: The generated sentences that have been post-processed.
        """
        generated_sentences = []
        if type(input_lengths)==int:
            input_lengths = [input_lengths] * len(output_sequences)
        for sent_id, seq in enumerate(output_sequences):
            seq = seq[input_lengths[sent_id]:]
            text_output = self.tokenizer.decode(seq, clean_up_tokenization_spaces=True)
            idx = text_output.find(self.tokenizer.eos_token)
            if idx >= 0:
                text_output = text_output[:idx]
            text_output = text_output.strip()
            generated_sentences.append(text_output)
        print(generated_sentences)
        return generated_sentences


    
    def prepare_inputs_for_generation(self, input_ids: Optional[torch.Tensor] = None,
                                         **model_kwargs):
        r"""This function wraps the `prepare_inputs_for_generation` function in the huggingface transformers.
        When the `past` not in model_kwargs, we prepare the input from scratch. 
        When `past` is in model_kwargs, we don't need to prepare the template wrapped input,
        instead we use the inner pretrain_models' function to prepare the next step's input.
        `model_kwargs` includes all the argument passed in the `batch`: InputFeatures, except `input_ids`
        , as long as they do not conflict with keywords in ``generation_kwargs``.    if 'past' not in model_kwargs: # the past_key_value not in model_kwargs, then we need to prepare input from scrath
        , as long as they do not conflict with keywords in ``generation_kwargs''.
        Args:
            input_ids(:obj:`torch.Tensor`): Indices of input sequence tokens in the vocabulary.
        """
        if self.generate_ith_token == 0 and 'encoder_outputs' not in model_kwargs: # generating the first token in decoder only setting.

            batch = InputFeatures(input_ids=input_ids, **model_kwargs)
            model_inputs = self.prompt_model.prepare_model_inputs(batch)
            # TODO check the competibility for more models. Having checked gpt2, T5
        else: # generating the subsequence generation can use the default setting
            model_inputs = self.plm.prepare_inputs_for_generation(input_ids, **model_kwargs)
        self.last_model_inputs = model_inputs  # to update the model_kwargs in _update_model_kwargs_for_generation, in-place operation.
        return model_inputs
    
    
    def _update_model_kwargs_for_generation(self,
        outputs, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        r""" The parents class's ``_update_model_kwargs_for_generation`` method will
        add past_key_values to model_kwargs, and update ``token_type_ids``, and ``attention_mask_ids``.
        In case some of the model_kwargs are modified in the prepare_inputs_for_generation function
        and should be used as the subsequent model_kwargs, we upate these kwargs before the parent class
        call. 
        Other updates should be added here after the parent's function call.
        Args:
            outputs (:obj:`torch.Tensor`): 
            is_encoder_decoder (:obj:`bool`, defaults to False): 
        """
        if self.generate_ith_token == 0:
            for key in self.last_model_inputs:
                if key in model_kwargs:
                    model_kwargs[key] = self.last_model_inputs[key]
        model_kwargs = super(PromptForGeneration, PromptForGeneration)._update_model_kwargs_for_generation(outputs=outputs, model_kwargs=model_kwargs, is_encoder_decoder=is_encoder_decoder)
        self.generate_ith_token += 1
        return model_kwargs


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        r""" This function resemble the function in GeneraionMix
        
        Args:
            input_ids (:obj:`torch.LongTensor`) The input ids for 
        """
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.plm.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }
            batch = {"input_ids":input_ids, **encoder_kwargs}
            model_inputs = self.prompt_model.prepare_model_inputs(batch) # This line differs from the orinigal code base, we should process the input
            # with our template, then pass it into the model.
            # some of the arguments may have been changed by the template,
            # e.g. the attention mask. Here we update the model_kwargs
            for key in model_kwargs:
                if key in model_inputs:
                    model_kwargs[key] = model_inputs[key]
            model_kwargs["encoder_outputs"] = encoder(return_dict=True, **model_inputs)
        return model_kwargs
    
    def state_dict(self):
        """ Save the model using template and plm's save methods. """
        _state_dict = {}
        if not self.prompt_model.freeze_plm:
            _state_dict['plm'] = self.plm.state_dict()
        _state_dict['template'] = self.template.state_dict()
        return _state_dict
    
    def load_state_dict(self, state_dict):
        """ Load the model using template and plm's load methods. """
        if 'plm' in state_dict and not self.prompt_model.freeze_plm:
            self.plm.load_state_dict(state_dict['plm'])
        self.template.load_state_dict(state_dict['template'])
    
    def _reorder_cache(self, past, beam_idx):
        r"""Use the plm's default _reorder_cache function
        """
        return self.plm._reorder_cache(past, beam_idx)

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")            

