"""
shared dataloader for multiple issues
must gurantee that a batch only has data from same issue
but the batches can be shuffled
"""

import collections
from more_itertools import more
from numpy.core import overrides
import torch
from torch import tensor
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from typing import *
from myprompt.data.example import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset, Dataset
from myprompt.utils.logging import logger
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from myprompt.template import Template
from myprompt.verbalizer import Verbalizer
from myprompt.plm.utils import TokenizerWrapper
from collections import defaultdict
from myprompt.utils.logging import logger
from myprompt.utils.utils import round_list, signature
from torch.utils.data.sampler import RandomSampler, Sampler
import random
import math

class MultiDataLoader(object):
    def __init__(self, dataloaders, shuffle_batch=True):
        super().__init__()
        self.dataloaders = dataloaders
        self.batches = sum([list(iter(self.dataloaders[k])) for k in self.dataloaders],[])
        if shuffle_batch:
            random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        return (b for b in self.batches)

class myDataset(Dataset):
    def __init__(self, data):
        self.samples = data

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)



class BatchSchedulerSampler(Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.samples) for cur_dataset in dataset.datasets])            

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            random_idx= list(range(self.number_of_datasets))
            random.shuffle(random_idx)
            #for i in range(self.number_of_datasets):
            for i in random_idx:
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)




class PromptShareDataLoader(object):
    def __init__(self, 
                 dataset: Dict[str, List],
                 template: Dict[str, Template],
                 tokenizer: PreTrainedTokenizer,
                 tokenizer_wrapper_class: TokenizerWrapper,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False, #shuffle_sample
                 shuffle_batch: Optional[bool] = True, #shuffle_batch
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 **kwargs,
                ):
        self.raw_dataset = dataset

        self.wrapped_dataset = collections.defaultdict(list)
        self.tensor_dataset = collections.defaultdict(list)
        self.template = template
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_batch = shuffle_batch
        self.teacher_forcing = teacher_forcing

        tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args 
        prepare_kwargs = {
            "max_seq_length" : max_seq_length,
            "truncate_method" : truncate_method,
            "decoder_max_length" : decoder_max_length,
            "predict_eos_token" : predict_eos_token,
            "tokenizer" : tokenizer,
            **kwargs,  
        }
        to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}

        self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)

        #check the wrap function
        for k in self.template:
            assert hasattr(self.template[k], 'wrap_one_example'), "Your prompt template has no function variable \
                                                         named wrap_one_example"        
        
        #process: 2 main steps of dataloader
        self.wrap()
        self.tokenize()

        concat_dataset = ConcatDataset([myDataset(self.tensor_dataset[k]) for k in self.tensor_dataset])
        self.dataloader = DataLoader(
            concat_dataset,
            batch_size = self.batch_size,
            sampler = BatchSchedulerSampler(concat_dataset, batch_size=batch_size),
            collate_fn = InputFeatures.collate_fct
        )        


    
    def wrap(self):
        """
        wrap the text with template
        """
        if isinstance(self.raw_dataset, Dict):
            for k in self.raw_dataset:
                for idx, example in enumerate(self.raw_dataset[k]):
                    wrapped_example = self.template[k].wrap_one_example(example)
                    self.wrapped_dataset[k].append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self):
        """
        Pass the wraped text into a prompt-specialized tokenizer
        """
        for k in self.wrapped_dataset:
            for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset[k]), desc ='tokenizing'):
                inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
                self.tensor_dataset[k].append(inputfeatures)
    
    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.dataloader.__iter__()    


