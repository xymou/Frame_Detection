import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from myprompt.data import InputExample, PromptDataLoader, PromptShareDataLoader
from myprompt.plm import load_plm
from myprompt.template import PtuningTemplate
from myprompt.verbalizer import LabelEmbedVerbalizer
from myprompt.model import PromptForSharedMLMTemplate
from transformers import  AdamW, get_linear_schedule_with_warmup
import re
import pandas as pd
import random
import pickle
from sklearn.metrics import f1_score,accuracy_score 
import copy
import sys
sys.path.append("../../data/")
from load_data_multi import *
from util import *
import collections
from metrics import *
import warnings
warnings.filterwarnings("ignore")

issue_label_words = {
    0:['climate'],
    1:['death','penalty'],
    2:['gun'],
    3:['immigration'],
    4:['gay','same-sex'],
    5:['tobacco'],
    6:['obama','care'],
    7:['abortion'],
    8:['terrorism']    
}

def pretrain_prompt():
    print('=============Prepare data from mfc, twitter, immi=============')
    mfc_data =read_mfc(data_type='article',issue_prepare=True)
    #twitter_data = read_twitter(issue_prepare=True) 
    immi_data = read_immi(issue='issue_generic', issue_prepare=True)
    data = {'text':mfc_data['text'], 'label':mfc_data['label'],
             'issue':mfc_data['issue']}
    data['label'] = convert_to_one_hot(data['label'],14)
    issues = list(set(data['issue']))
    issue_data = {i:collections.defaultdict(list) for i in issues}
    for i in range(len(data['label'])):
        issue_data[data['issue'][i]]['text'].append(data['text'][i])
        issue_data[data['issue'][i]]['label'].append(data['label'][i])
        issue_data[data['issue'][i]]['issue'].append(data['issue'][i])
    
    classes = load_obj('mfc_classes')
    label_text = load_obj('mfc_labelname')
    print(label_text)
    dataset = {'train':collections.defaultdict(list), 'validation':collections.defaultdict(list), 'test':collections.defaultdict(list)}
    j=0
    for k in issue_data:
        train, dev, test = split_data(issue_data[k], issue_prepare=True)
        for i in range(len(train['text'])):
            input_example = InputExample(text_a = train['text'][i],  label=train['label'][i], guid=j, issue=int(train['issue'][i]))
            dataset['train'][k].append(input_example)
            j+=1
        for i in range(len(dev['text'])):
            input_example = InputExample(text_a = dev['text'][i],  label=dev['label'][i], guid=j, issue=int(dev['issue'][i]))
            dataset['validation'][k].append(input_example)
            j+=1
        for i in range(len(test['text'])):
            input_example = InputExample(text_a = test['text'][i],  label=test['label'][i], guid=j, issue=int(test['issue'][i]))
            dataset['test'][k].append(input_example)
            j+=1   
    
    print('=============Prepare Model=============')
    plm, tokenizer, model_config, WrapperClass = load_plm('bertmlm', '/bert/bert-base-uncased/')
    template_text = '{"placeholder":"text_a"} {"soft":"It"} {"soft":"emphasizes"} {"mask"} {"soft":"aspect"}.'
    #template_text = '{"soft":None, "duplicate":50} {"placeholder":"text_a"} {"soft":None, "duplicate":5} {"mask"}.'
    mytemplate = {k:PtuningTemplate(tokenizer=tokenizer, text=template_text, model=plm, prompt_encoder_type="lstm") for k in issue_data}
    mytemplate['share'] = PtuningTemplate(tokenizer=tokenizer, text=template_text, model=plm, prompt_encoder_type="lstm")
    
    train_dataloader = PromptShareDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3, 
        batch_size=8,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method='head')
    validation_dataloader = PromptShareDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3, 
        batch_size=8,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method='head')
    test_dataloader = PromptShareDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3, 
        batch_size=8,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method='head')
    # verbalizer
    myverbalizer = LabelEmbedVerbalizer(
        classes = classes,
        label_text= label_text,
        tokenizer = tokenizer,
        plm=plm
    )
    issue_verbalizer = LabelEmbedVerbalizer(
        classes = list(range(6)),
        label_text={k:v for k,v in issue_label_words.items() if k in list(range(6))},
        tokenizer =tokenizer,
        plm = plm
    )
    use_cuda = True
    device=torch.device('cuda:{}'.format(1))
    prompt_model =PromptForSharedMLMTemplate(plm=plm, template=mytemplate, verbalizer_label=myverbalizer, verbalizer_issue=issue_verbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model = prompt_model.to(device)  
    loss_func1 = nn.BCELoss()
    loss_func2 = nn.CrossEntropyLoss()
    
    no_decay = ['bias', 'LayerNorm.weight']
    # training
    print('==============Start Training=============')
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]       
    optimizer_grouped_parameters2 = []
    for k in mytemplate:
        optimizer_grouped_parameters2.append({'params': [p for n,p in getattr(prompt_model.prompt_model,'template_'+str(k)).named_parameters() if "raw_embedding" not in n]})
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=2e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=2e-1)
    best_score = 0
    patience_counter = 0
    patience = 5
    epochs = 20
    lamb= 0.05
    gradient_accumulation_steps=2
    saved_model = torch.load('../task1/pivot_encoder_top50.pkl', map_location=torch.device('cpu'))
    state_dict={}
    for key in saved_model:
        if key.startswith('bert'):
            state_dict[key[5:]]=saved_model[key]
    for key in prompt_model.plm.state_dict():
        if key not in state_dict:
            print('missing: ', key)
            state_dict[key] = prompt_model.plm.state_dict()[key]
    prompt_model.plm.load_state_dict(state_dict)
    print('Successfully initialize plm!')

    for epoch in range(epochs):
        tot_loss = 0 
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.to(device)
            p = float(step+epoch * len(train_dataloader)) / epochs / len(train_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            prompt_model.train()
            issue = inputs['issue'][0].item()
            logits1, logits2, out1, out2 = prompt_model(inputs,issue, alpha)
            labels = inputs['label'].squeeze(1)
            issues = inputs['issue']
            loss1 = loss_func1(logits1, labels)
            loss2 = loss_func2(logits2, issues)
            print('loss1: ',loss1.item(), ' loss2: ', loss2.item())
            loss = loss1+ loss2
            loss.backward()
            tot_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer1.step()
                optimizer2.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
        print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
        f1_micro, f1_macro = eval_multi_share(prompt_model, validation_dataloader)
        print('epoch metrics on validation set:f1_micro:{}, f1_macro:{}'.format(f1_micro, f1_macro))
        #early stopping
        if f1_micro < best_score:
            patience_counter += 1
        else:
            best_score= f1_micro
            patience_counter=0
            best_model=PromptForSharedMLMTemplate(plm=plm,template=mytemplate, verbalizer_label=myverbalizer, verbalizer_issue=issue_verbalizer, freeze_plm=True)
            best_model.load_state_dict(copy.deepcopy(prompt_model.state_dict()))
        if patience_counter >=patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break                        

    # evaluate and save best model
    print('==============Evaluate on test set=============')
    best_model.eval()
    f1_micro, f1_macro=eval_multi_share(prompt_model, test_dataloader)
    print('metrics on test set: f1_micro:{}, f1_macro:{}'.format(f1_micro, f1_macro))   
    print('==============Save best model=============')  
    torch.save(best_model.state_dict(),'pretrain_param_top50.pkl')  


if __name__=='__main__':
    pretrain_prompt()
