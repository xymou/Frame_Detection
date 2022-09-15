import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import transformer
from tqdm import tqdm
from transformers import  AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from model import BertForMaskedLM
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
import re
import pandas as pd
from tqdm import tqdm,trange
import random
import pickle
from sklearn.metrics import f1_score,accuracy_score 
import copy
import sys
sys.path.append("../../data/")
from load_data_multi import *
from torch.utils.data import Dataset, DataLoader
import collections
import os
import warnings
warnings.filterwarnings("ignore")



class myDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, pivot2id, pivot_prob, non_pivot_prob):
        # store encodings internally
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pivot2id = pivot2id
        self.pivot_prob = pivot_prob
        self.non_pivot_prob = non_pivot_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data[i]
        tokens = self.tokenizer.tokenize(text)
        if len(tokens)>self.max_len-2:  #trucated
            tokens = tokens[: self.max_len-2]
        tokens, labels = random_mask_word(tokens, self.pivot2id, self.pivot_prob, self.non_pivot_prob)
        lm_labels = ([-1]+ labels+ [-1])
        tokens = ['CLS']+ tokens + ['SEP']
        segment_ids= [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        #zero-padding
        while len(input_ids)< self.max_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_labels.append(-1)
        assert len(input_ids) == self.max_len
        
        cur_tensors = (torch.tensor(input_ids),
                       torch.tensor(input_mask),
                       torch.tensor(segment_ids),
                       torch.tensor(lm_labels))

        return cur_tensors

from nltk.stem.snowball import SnowballStemmer
import string
stemmer=SnowballStemmer('english')
def random_mask_word(tokens, pivot2id, pivot_prob=0.5, non_pivot_prob=0.1):
    output_label = []
    for i, token in enumerate(tokens):
        prob = random.random()
        token = stemmer.stem(token)
        if token in pivot2id:
            if prob< pivot_prob:
                prob /= pivot_prob
                if prob<0.8:
                    tokens[i] = "[MASK]"
                    output_label.append(pivot2id[token])
                    continue
        elif prob<non_pivot_prob:
            prob /=0.1
            if prob < 0.8:
                tokens[i] = "[MASK]"
            output_label.append(0)
            continue
        output_label.append(-1)
    return tokens, output_label


def main():
    ### prepare data, mask pivots
    print('=============Prepare data from mfc=============')
    mfc_data =read_mfc(data_type='article',issue_prepare=True)
    data = mfc_data['text']
    pivots = load_obj('mfc_new_words_top50')
    pivots = sum(list(pivots.values()),[])
    pivots = list(set(pivots))
    pivot2id={}
    id2pivot={}
    pivot2id['NONE'] = 0
    id2pivot[0] = 'NONE'
    for id, pi in enumerate(pivots):
        pivot2id[pi] = id+1
        id2pivot[id+1] = pi
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    tokenizer = BertTokenizer.from_pretrained('/remote-home/xymou/bert/bert-base-uncased/')
    dataset = myDataset(data, tokenizer, max_len=512, pivot2id=pivot2id, pivot_prob=args.pivot_prob, non_pivot_prob=args.non_pivot_prob)
    dataloader = DataLoader(dataset, batch_size =args.train_batch_size, shuffle=True)
    # define model
    num_train_optimization_steps = int(
            len(dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    model = BertForMaskedLM.from_pretrained(args.bert_model, output_dim=len(pivot2id))
    device=torch.device('cuda:{}'.format(args.cuda)) 
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]    
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion, t_total=num_train_optimization_steps)

    # freeze all bert weights, train only last encoder layer
    try:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for id, param in enumerate(model.bert.encoder.layer.parameters()):
            if id < (192 * (12-args.num_of_unfrozen_bert_layers) / 12):
                param.requires_grad = False
        for param in model.cls.predictions.pivots_decoder.parameters():
            param.requires_grad = args.train_output_embeds
    except:
        for param in model.module.bert.embeddings.parameters():
            param.requires_grad = False
        for id, param in enumerate(model.module.bert.encoder.layer.parameters()):
            if id < (192 * (12 - args.num_of_unfrozen_bert_layers) / 12):
                param.requires_grad = False
        for param in model.module.cls.predictions.pivots_decoder.parameters():
            param.requires_grad = args.train_output_embeds    


    # loss_func = nn.CrossEntropyLoss()
    ### train and save model
    print('==============Start Training=============')
    model.train()
    for epoch in trange(int(args.num_train_epochs), desc = 'Epoch'):
        tot_loss= 0 
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, lm_label_ids)
            tot_loss += loss.item()
            loss.backward()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if step % 50==1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

    torch.save(model.state_dict(),'pivot_encoder_top50.pkl')  


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default='/bert/bert-base-uncased/', type=str)
    parser.add_argument("--pivot_prob",default=0.5,type=float)
    parser.add_argument("--non_pivot_prob",default=0.1,type=float)
    parser.add_argument("--num_train_epochs",default=20.0,type=float)      
    parser.add_argument("--train_batch_size",default=8, type=int)   
    parser.add_argument("--lr",default=3e-5,type=float)                       
    parser.add_argument("--warmup_proportion",default=0.1,type=float)
    parser.add_argument('--num_of_unfrozen_bert_layers',type=int,default=8)    
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1)
    parser.add_argument('--init_output_embeds',action='store_true')
    parser.add_argument('--train_output_embeds', action='store_true')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])    
    main()
