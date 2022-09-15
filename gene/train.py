import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from myprompt.data import InputExample, PromptDataLoader
from myprompt.plm import load_plm
from myprompt.template import  PtuningTemplate
from myprompt.verbalizer import  LabelEmbedVerbalizer
from myprompt.model import PromptForLabelMatching
from transformers import AdamW, get_linear_schedule_with_warmup
import re
import pandas as pd
import random
import pickle
from sklearn.metrics import f1_score,accuracy_score 
import copy
from load_data_multi import *
from util import *
import collections
from metrics import *
import warnings
warnings.filterwarnings("ignore")
import logging

def main():
    args['freeze_plm'] = bool(args['freeze_plm'])
    print('=============Prepare data from {}============='.format(args['dataset']))
    data_config={
        'dataset':args['dataset'],
        'issue':args['issue'],
        'data_type':args['data_type']
    }
    data=read_data(data_config)   
    data['label'] = convert_to_one_hot(data['label'], args['num_class'])
    print('Size of the dataset: ', len(data['text'])) 
    classes = load_obj(args['frame_classes'])
    label_text = load_obj(args['label_text'])
    print('# of frame classes: ', len(classes))
    assert len(classes) == args['num_class']

    if args['few_shot']>=0:
        train_data, dev_data, test = split_data(data)
        train_data['text'] += dev_data['text']
        train_data['label'] += dev_data['label']
        train, dev = split_data_for_fewshot(train_data,args['num_class'], args['few_shot'])
    else:
        train, dev, test=split_data(data)
    print('dataset size: ',len(train['label']), len(dev['label']), len(test['label']))
    raw_dataset={'train':train, 'validation':dev, 'test':test}
    dataset = {}
    j=0
    for split in ['train', 'validation', 'test']:
        dataset[split] = []
        for i in range(len(raw_dataset[split]['text'])):
            input_example = InputExample(text_a = raw_dataset[split]['text'][i],  label=raw_dataset[split]['label'][i], guid=j)
            dataset[split].append(input_example)
            j+=1    
    print('=============Prepare Model=============')
    #prepare PLM and Template
    plm, tokenizer, model_config, WrapperClass = load_plm(args['model'], args['model_name_or_path'])
    template_text = '{"placeholder":"text_a"} {"soft":"It"} {"soft":"emphasizes"} {"mask"} {"soft":"aspect"}.'
    mytemplate = PtuningTemplate(tokenizer=tokenizer, text=template_text, model=plm, prompt_encoder_type="lstm")

    #dataloader
    args['train_batchsize'] = args['train_batchsize']//args['gradient_accumulation_steps']
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args['max_length'], decoder_max_length=3, 
        batch_size=args['train_batchsize'],shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method=args['truncate'])
    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args['max_length'], decoder_max_length=3, 
        batch_size=args['dev_batchsize'],shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method=args['truncate'])
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args['max_length'], decoder_max_length=3, 
        batch_size=args['test_batchsize'],shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method=args['truncate'])
    
    # verbalizer
    myverbalizer = LabelEmbedVerbalizer(
        classes = classes,
        label_text= label_text,
        tokenizer = tokenizer,
        plm=plm
    )

    use_cuda = args['cuda_available']
    device=torch.device('cuda:{}'.format(args['cuda'])) 
    prompt_model = PromptForLabelMatching(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=args['freeze_plm'])
    if use_cuda:
        prompt_model = prompt_model.to(device)  
    loss_func = torch.nn.BCELoss()       
    no_decay = ['bias', 'LayerNorm.weight']
    # training
    print('==============Start Training=============')
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]  
    optimizer_grouped_parameters2 = [{'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args['lr1'])
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args['lr2'])
    best_score = 0
    best_loss = float("inf")
    patience_counter = 0
    patience = 5
    gradient_accumulation_steps=args['gradient_accumulation_steps']

    saved_model = torch.load('pretrain_param_top50.pkl',  map_location=torch.device('cpu'))
    state_dict = copy.deepcopy(saved_model['template_share']) 
    prompt_model.prompt_model.template.load_state_dict(state_dict)
    prompt_model.plm.load_state_dict(saved_model['plm'])
    print('Successfully initialize prompt!')

    for epoch in range(args['epochs']):
        tot_loss = 0 
        for step, inputs in enumerate(train_dataloader):
            
            if use_cuda:
                inputs = inputs.to(device)
            prompt_model.train()
            logits = prompt_model(inputs)
            labels = inputs['label'].squeeze(1)
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer1.step()
                optimizer2.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
        print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        f1_micro, f1_macro, p_at_1, p_at_3, loss = eval_multi(prompt_model, validation_dataloader, loss_func=loss_func)
        print('epoch metrics on validation set:f1_micro:{}, f1_macro:{}, p_at_1:{}, p_at_3:{}'.format(f1_micro, f1_macro,  p_at_1, p_at_3), loss)

        #early stopping
        if f1_micro < best_score:
            patience_counter += 1
        else:
            best_score= f1_micro
            patience_counter=0
            best_model=PromptForLabelMatching(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=args['freeze_plm'])
            best_model.load_state_dict(copy.deepcopy(prompt_model.state_dict()))
        if patience_counter >=patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break           

    # evaluate
    print('==============Evaluate on test set=============')
    best_model.to(device)
    best_model.eval()
    thred = search_thres_by_f1(best_model, validation_dataloader)
    print('best thred: ', thred)
    f1_micro, f1_macro, p_at_1, p_at_3, _=eval_multi(best_model, test_dataloader, thred= thred)
    print('metrics on test set: f1_micro:{}, f1_macro:{}, p_at_1:{}, p_at_3:{}'.format(f1_micro, f1_macro,  p_at_1, p_at_3))


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d','--dataset', default='gvfc')
    parser.add_argument('-i','--issue', default='all')
    parser.add_argument('-fc','--frame_classes', default='gvfc_classes')
    parser.add_argument('-lw','--label_text', default='gvfc_labelname')    
    parser.add_argument('-dt','--data_type', default='article')    
    parser.add_argument('-tb','--train_batchsize', type=int, default=16)
    parser.add_argument('-db','--dev_batchsize', type=int, default=16)
    parser.add_argument('-teb','--test_batchsize', type=int, default=16)
    parser.add_argument('-g','--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('-e','--epochs', type=int, default=25)
    parser.add_argument('-lr1','--lr1', type=float, default=2e-5)
    parser.add_argument('-lr2','--lr2', type=float, default=1e-2)
    parser.add_argument('-ca','--cuda_available', default=True)
    parser.add_argument('-c','--cuda', default=0)
    parser.add_argument('-m','--model', default='bertmlm')
    parser.add_argument('-p','--model_name_or_path', default='/bert/bert-base-uncased/')
    parser.add_argument('-ml','--max_length', type=int,  default=128)
    parser.add_argument('-tr','--truncate', default='head')
    parser.add_argument('-fr','--freeze_plm', type=int, default=0)
    parser.add_argument('-n','--num_class', type=int, default=9)
    parser.add_argument('-f','--few_shot', type=int, default=-1)
    parser.add_argument('-r','--random_seed', type=int, default=42)

    args = vars(parser.parse_args())
    random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    main()    
