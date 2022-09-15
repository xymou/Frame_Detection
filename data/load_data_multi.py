import numpy as np
import pandas as pd
import torch
import pickle
import random
import re
from tqdm import tqdm
from transformers.modeling_utils import PoolerAnswerClass

data_path={
    'mfc': '/data/news/mfc/',
    'gvfc': '/data/news/GVFC/GVFC/GVFC_headlines_and_annotations.xlsx',
    'twitter': '/data/tweet/twitter/',
    'immi': '/data/tweet/immi/',
    'fora': '/data/debate/issue_framing/data/dialogue/'
}

all_issue_map={
    'climate':0,
    'deathpenalty':1,
    'guncontrol':2,
    'immigration':3,
    'samesex':4,
    'tobacco':5,
    'aca':6,
    'abort':7,
    'immig':3,
    'isis':8,
    'guns':2,
    'lgbt':4    

}


def save_obj(obj, name):
    with open('obj' + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name,path=None):
    if path is None:
        with open('obj' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        with open(path+'obj' + name + '.pkl', 'rb') as f:
            return pickle.load(f) 

def clean(text):
    pattern1='(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    pattern2='@([^@ ]*)'
    pattern3='pic.twitter.com/.*'
    text=re.sub(pattern1,' [URL]',text)
    text=re.sub(pattern2,' [MENTION]',text)
    text=re.sub(pattern3,' [PIC]',text)
    text=re.sub('\xa0','',text)
    return text  



def split_data(data, issue_prepare=False): 
    n=len(data['text'])
    idx=list(range(n))
    random.shuffle(idx)
    if issue_prepare:
        train,dev,test={"text": [], "label": [],'issue':[]},{"text": [], "label": [],'issue':[]},{"text": [], "label": [],'issue':[]}
    else:
        train,dev,test={"text": [], "label": []},{"text": [], "label": []},{"text": [], "label": []}
    for i in idx[:int(0.7*n)]:
        train['text'].append(data['text'][i])
        train['label'].append(data['label'][i])
        if issue_prepare:
            train['issue'].append(data['issue'][i])
    for i in idx[int(0.7*n):int(0.9*n)]:
        dev['text'].append(data['text'][i])
        dev['label'].append(data['label'][i])  
        if issue_prepare:
            dev['issue'].append(data['issue'][i])      
    for i in idx[int(0.9*n):]:
        test['text'].append(data['text'][i])
        test['label'].append(data['label'][i])      
        if issue_prepare:
            test['issue'].append(data['issue'][i])    
    return train,dev,test


def split_data_for_fewshot(data, num_class, shot=1):
    n=len(data['text'])
    idx=list(range(n))
    random.shuffle(idx)
    train,dev,test={"text": [], "label": []},{"text": [], "label": []},{"text": [], "label": []}
    label_set=list(range(num_class))
    used=[]
    
    label_dict={l:[] for l in label_set}
    for i in idx:
        for j in range(num_class):
            if data['label'][i][0][j]==1:
                label_dict[j].append(i)
    print({k:len(label_dict[k]) for k in label_dict})
    for l in label_dict:
        if len(label_dict[l])>shot:
            k,j=0,0
            while k<shot:
                if j<len(label_dict[l]) and label_dict[l][j] not in used:
                    train['text'].append(data['text'][label_dict[l][j]])
                    train['label'].append(data['label'][label_dict[l][j]])
                    used.append(label_dict[l][j])
                    k+=1
                    j+=1
                else:
                    j+=1
                    if j>len(label_dict[l]):break
        else:
            for j in range(len(label_dict[l])):
                if label_dict[l][j] not in used:
                    train['text'].append(data['text'][label_dict[l][j]])
                    train['label'].append(data['label'][label_dict[l][j]])
                    used.append(label_dict[l][j])                            
        if len(label_dict[l])>2*shot:
            k,j=0,0
            while k<shot:
                if j<len(label_dict[l]) and label_dict[l][j] not in used:
                    dev['text'].append(data['text'][label_dict[l][j]])
                    dev['label'].append(data['label'][label_dict[l][j]])
                    used.append(label_dict[l][j])
                    k+=1
                    j+=1
                else:
                    j+=1
                    if j>len(label_dict[l]):break
        else:
            for j in range(len(label_dict[l])):
                if label_dict[l][j] not in used:
                    dev['text'].append(data['text'][label_dict[l][j]])
                    dev['label'].append(data['label'][label_dict[l][j]])
                    used.append(label_dict[l][j])                    
    return train,dev 




def read_mfc_issue(path='/remote-home/xymou/Frame/framework/data/news/mfc/', data_type='article', issue='climate'):
    if data_type=='article':
        data=load_obj('article_data_multi',path)     
    elif data_type=='sentence':
        data=load_obj('sentence_data_multi',path)
    else:
        raise Exception('Undefined data type! Choose from [article, sentence]')
    return data[issue]


def read_mfc(path='/remote-home/xymou/Frame/framework/data/news/mfc/', data_type='article', issue='all',issue_prepare=False):
    print('Reading data from MFC dataset!')
    if issue!='all':
        issues= [issue]
    else:
        issues= ['climate', 'deathpenalty', 'guncontrol', 'immigration', 'samesex', 'tobacco']
    if issue_prepare:
        data = {'text':[], 'label':[], 'issue':[]}
    else:
        data = {'text':[], 'label':[]}
    for i in issues:
        tmp = read_mfc_issue(path, data_type, issue=i)
        data['text'].extend(tmp['text'])
        data['label'].extend(tmp['label'])  
        if issue_prepare: 
            data['issue'].extend([all_issue_map[i]]*len(tmp['label']))  
    return data

def read_gvfc(path='/remote-home/xymou/Frame/framework/data/news/GVFC/GVFC/GVFC_headlines_and_annotations.xlsx'):
    print('Reading data from GVFC dataset!')
    df=pd.read_excel(path)
    data,label=[],[]
    for i in tqdm(range(len(df))):
        text=df.loc[i,'news_title']   
        if df.loc[i,'Q1 Relevant']==1 and df.loc[i,'Q3 Theme1']!=99:
            data.append(text.lower()) 
            tmp=[df.loc[i,'Q3 Theme1']-1]
            if df.loc[i,'Q3 Theme2']!=99:
                tmp.append(df.loc[i,'Q3 Theme2']-1)
            label.append(tmp)       
    return {"text": data, "label": label}    

def read_twitter_issue(path='/remote-home/xymou/Frame/sample_test/Weakly/', issue='aca'):
    tweet_processed=load_obj('tweet_processed', path)
    label_map={
        0:0,1:1,2:2,3:3,4:4,5:6,6:7,7:8,8:9,9:10,10:11,11:12,12:5,13:13
    }    
    text,label=[],[]
    for key in tweet_processed:
        if issue in tweet_processed[key]['issue']:
            tmp=tweet_processed[key]['text']
            tmp=clean(tmp.lower())
            tmp=re.sub('\#','',tmp)  
            res = [label_map[k-1] for k in tweet_processed[key]['frame'] if k not in [15,16,17]]
            if len(res):   
                label.append(res)      
                text.append(tmp)
    return {'text':text,'label':label}    


def read_twitter(path='/remote-home/xymou/Frame/sample_test/Weakly/', issue='all', issue_prepare = True):
    print('Reading data from Twitter-framing dataset!')
    if issue == 'all':
        issues=['aca','abort','immig','isis','guns','lgbt']
    else:
        issues = [issue]
    if issue_prepare:
        data = {'text':[], 'label':[], 'issue':[]}
    else:
        data = {'text':[], 'label':[]}
    for i in issues:
        tmp = read_twitter_issue(path,  i)
        for j in range(len(tmp['label'])):
            data['text'].append(tmp['text'][j])
            data['label'].append(tmp['label'][j])    
            if issue_prepare:
                data['issue'].append(all_issue_map[i])  
    return data



def read_immi(path='/remote-home/xymou/Frame/framework/data/tweet/immi/' , issue='issue_specific', issue_prepare=False): #这里的issue其实是ftype
    print('Reading data from immigration twitter dataset!')
    text, label = [],[]
    data = load_obj(issue, path)
    if issue=='issue_generic':
        labels = ['Cultural Identity','Capacity and Resources','Security and Defense','Quality of Life',
    'Crime and Punishment','Policy Prescription and Evaluation','Morality and Ethics','External Regulation and Reputation','Health and Safety',
    'Political Factors and Implications','Public Sentiment','Economic','Fairness and Equality','Legality, Constitutionality, Jurisdiction'    ]
    else:
        labels = ['Victim: Global Economy','Threat: Fiscal', 'Hero: Cultural Diversity', 'Threat: Public Order', 'Threat: Jobs',
    'Victim: Humanitarian', 'Threat: National Cohesion','Hero: Integration','Victim: Discrimination','Victim: War','Hero: Worker']
    label_map={l:labels.index(l) for l in labels}
    for i in range(len(data['text'])):
        text.append(clean(data['text'][i].lower()))
        label.append([label_map[k] for k in data['label'][i]])
    if issue == 'issue_generic' and issue_prepare:
        return {'text':text, 'label':label, 'issue':[all_issue_map['immigration']]*len(label)}
    return {'text':text,'label':label}


def read_fora(path='/remote-home/xymou/Frame/framework/data/debate/issue_framing/data/dialogue/'):
    print('Reading data from Fora dataset!')
    text, label=[],[]
    data = load_obj('fora_data',path)
    for i in range(len(data['label'])):
        text.append(data['text'][i])
        label.append(data['label'][i])
    return {'text':text, 'label':label}



data_func_map={
    'mfc':read_mfc,
    'gvfc':read_gvfc,
    'twitter':read_twitter,
    'immi':read_immi,
    'fora':read_fora    

}
def read_data(config):
    dataset = config['dataset']
    if dataset not in data_func_map:
        raise KeyError('Current dataset is not mapped to data_read function! Please define the read data function for this dataset!')
    func = data_func_map[dataset]
    config.pop('dataset')
    if dataset != 'mfc':
        config.pop('data_type')
    if dataset in ['gvfc','fora']:
        config.pop('issue')
    config['path'] = data_path[dataset]
    return func(**config)


def convert_to_one_hot(label, label_num):
    print('# of labels:', label_num)
    res= []
    for l in label:
        tmp=[0]*label_num
        for i in range(len(l)):
            tmp[l[i]]=1
        res.append(torch.tensor(tmp, dtype=torch.float32).view(1,-1))
    return res

  
from torch.utils.data import Dataset
class mydata(Dataset):
    def __init__(self, data, tokenizer, padding_idx=0, max_len=None):
        self.text_lengths=[len(seq) for seq in data['text']]
        self.max_len=max_len
        if self.max_len is None:
            self.max_len=max(self.text_lengths)
        self.num_sequences = len(data["text"])
        self.data=data
        for i in range(len(data['text'])):
            data['text'][i] = tokenizer.encode(data['text'][i], max_length=self.max_len)
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, index):
        return {"text":self.data['text'][index],
                 "label":self.data['label'][index],
                 "issue":self.data['issue'][index]}
