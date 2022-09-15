import torch
import torch.nn as nn
from tqdm import tqdm
from myprompt.data.example import InputExample
import re
import pandas as pd
import random
from sklearn.metrics import f1_score,accuracy_score 
import time
import copy
from myprompt.plm import load_plm
from myprompt.template import ManualTemplate
from myprompt.data import PromptDataLoader
from myprompt.verbalizer import ManualVerbalizer
from myprompt.model import PromptForClassification
from transformers import  AdamW, get_linear_schedule_with_warmup

random.seed(1207)


def read_data(path='/remote-home/xymou/Frame/framework/data/news/GVFC/GVFC/GVFC_headlines_and_annotations.xlsx'):
    df=pd.read_excel(path)
    data,label=[],[]
    for i in tqdm(range(len(df))):
        text=df.loc[i,'news_title']   
        if df.loc[i,'Q1 Relevant']==1 and df.loc[i,'Q3 Theme1']!=99:
            data.append(text.lower()) 
            label.append(df.loc[i,'Q3 Theme1']-1)       
    return {"text": data, "label": label}

def split_data(data): #7,2,1
    n=len(data['text'])
    idx=list(range(n))
    random.shuffle(idx)
    train,dev,test={"text": [], "label": []},{"text": [], "label": []},{"text": [], "label": []}
    for i in idx[:int(0.7*n)]:
        train['text'].append(data['text'][i])
        train['label'].append(data['label'][i])
    for i in idx[int(0.7*n):int(0.9*n)]:
        dev['text'].append(data['text'][i])
        dev['label'].append(data['label'][i])        
    for i in idx[int(0.9*n):]:
        test['text'].append(data['text'][i])
        test['label'].append(data['label'][i])      
    return train,dev,test 

def cal_acc(prompt_model,dataloader):
    use_cuda = True
    device=torch.device('cuda:0')
    prompt_model.eval()
    with torch.no_grad():
        allpreds = []
        alllabels = []
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return acc   


def main():
    # 1.prepare data
    classes = [
        'Gun/2nd Amendment rights',
        'Gun control/regulation',
        'Politics',
        'Mental health',
        'School or public space safety',
        'Race/ethnicity',
        'Public opinion',
        'Society/culture',
        'Economic consequences'
        ]
    frame_map={classes[i]:i for i in range(len(classes))}
    data=read_data()
    train,dev,test=split_data(data)
    raw_dataset={'train':train, 'validation':dev, 'test':test}
    dataset = {}
    j=0
    for split in ['train', 'validation', 'test']:
        dataset[split] = []
        for i in range(len(raw_dataset[split]['text'])):
            input_example = InputExample(text_a = raw_dataset[split]['text'][i],  label=int(raw_dataset[split]['label'][i]), guid=j)
            dataset[split].append(input_example)
            j+=1   
    #print(dataset['train'][0])

    # 2.prepare PLM
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "/remote-home/xymou/bert/bert-base-uncased/")
     
    # 3.prepare template
    template_text = '{"placeholder":"text_a"} .It emphasizes {"mask"} aspect.'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)    

    #wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 
    #print(wrapped_example)    

    wrapped_berttokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer,truncate_method="head")

    #tokenized_example = wrapped_berttokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
    #print(tokenized_example)
    #print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))

    # 4.prepare dataloader
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3, 
        batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")    

    validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3, 
        batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")    

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=128, decoder_max_length=3, 
        batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")
    
    # 5.prepare verbalizer
    myverbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            'Gun/2nd Amendment rights':['constitution','liberty','ownership','rights',
                                    'democracy','defense'],
            'Gun control/regulation':['regulation','control','limit','licensing','elimination',
                                    'regulating','supervision'],
            'Politics':['politics','goverment','governance','republic',
                'politician','election','parliament','democratic'
                'campaign','republican','party','congress'],
            'Mental health':['mental','emotional','instability', 'impulsivity', 'anger'],
            'School or public space safety':['safety','police','teachers','monitoring','officer','measures',
                                    'detectors','backpacks'],
            'Race/ethnicity':['race','ethnic','immigration','border','muslim','terrorists',
                        'african','discrimination','revenge','black','white'],
            'Public opinion':['opinion','parade','protest','opinion','poll','emotion','mourning',
                            'walkout','activists','boycot'],
            'Society/culture':['society','culture','media','TV','movie','video','games'
                        'cliques','bullying','isolation','family','religious'],
            'Economic consequences':['finance','loss','gain','cost','sale','tax','revenue',
                                'manufacturing','money','budget','corporate','seller','shop']       
        },
        tokenizer = tokenizer,
    )    

    # 6. training prompt model
    use_cuda = True
    device=torch.device('cuda:0')
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    if use_cuda:
        prompt_model=  prompt_model.to(device)
    
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
    best_score = 0
    patience_counter = 0
    patience = 5
    print('-------------------------------------------')
    print('start training...')
    for epoch in range(10):
        tot_loss = 0 
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.to(device)
            prompt_model.train()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step %50 ==1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)
            
                epoch_acc=cal_acc(prompt_model, validation_dataloader)
                print('epoch acc on validation set: '+str(epoch_acc))
                #early stopping
                if epoch_acc < best_score:
                    patience_counter += 1
                else:
                    best_score= epoch_acc
                    patience_counter=0
                    best_model=PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
                    best_model.load_state_dict(copy.deepcopy(prompt_model.state_dict()))
                if patience_counter >=patience:
                    print("-> Early stopping: patience limit reached, stopping...")
                    break            

    print('-------------------------------------------')
    print('start evaluation...')
    best_model.eval()
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.to(device)
        logits = best_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    print('acc: ', accuracy_score(alllabels, allpreds))     
    print('macro_f1: ', f1_score(alllabels, allpreds, average='macro'))
    print('micro_f1: ', f1_score(alllabels, allpreds, average='micro'))
    
if __name__=='__main__':
    main()