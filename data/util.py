import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.stats
import collections

from metrics import F1Score
from metrics import *
def p_at_k(preds, labels, k=1):
    #按照prob排序
    #micro - 直接按样本平均
    preds_topk = torch.argsort(preds, dim = 1, descending=True)
    res = []
    for i in range(labels.size(0)):
        tmp = 0
        for j in range(k):
            if labels[i][preds_topk[i][j]] == 1:
                tmp+=1
        res.append(tmp/k)
    return np.mean(res)

def eval_multi(prompt_model, dataloader, use_cuda=True, thred=0.25, loss_func = nn.BCELoss()):
    f1_micro = F1Score(task_type='binary', average='micro',thresh=thred, normalizate=False)
    f1_macro = F1Score(task_type='binary', average='macro',thresh=thred, normalizate=False)

    device = prompt_model.device
    prompt_model.eval()
    alpha = 0
    loss=0
    with torch.no_grad():
        allpreds = []
        alllabels = []
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.to(device)
            logits = prompt_model(inputs)
            #logits = F.sigmoid(logits)
            labels = inputs['label'].squeeze(1)
            loss += loss_func(logits,labels).item()
            alllabels.append(torch.tensor(labels, dtype=torch.float32))
            allpreds.append(logits)
        alllabels=torch.cat(alllabels, dim=0)
        allpreds=torch.cat(allpreds, dim=0)
     
        f1_micro(allpreds, alllabels)
        f1_micro = f1_micro.value()
        f1_macro(allpreds, alllabels)
        f1_macro = f1_macro.value()  
        p_at_1 = p_at_k(allpreds, alllabels, k=1)
        p_at_3 = p_at_k(allpreds, alllabels, k=3)

      
        return f1_micro, f1_macro, p_at_1,p_at_3, loss
      
      
def search_thres_by_f1(model, dataloader, use_cuda=True):
    thres_ls = np.arange(1, 100) / 100
    f1_ls = []
    device = model.device
    for th in thres_ls:
        f1_micro = F1Score(task_type='binary', average='micro',thresh=th)
        f1_macro = F1Score(task_type='binary', average='macro',thresh=th)
        model.eval()
        with torch.no_grad():
            allpreds = []
            alllabels = []
            for step, inputs in enumerate(dataloader):
                if use_cuda:
                    inputs = inputs.to(device)
                logits = model(inputs)
                labels = inputs['label'].squeeze(1)
                alllabels.append(torch.tensor(labels, dtype=torch.float32))
                allpreds.append(logits)
            alllabels=torch.cat(alllabels, dim=0)
            allpreds=torch.cat(allpreds, dim=0)

            f1_micro(allpreds, alllabels)
            f1_micro = f1_micro.value()
            f1_macro(allpreds, alllabels)
            f1_macro = f1_macro.value() 

        f1_ls.append(f1_macro)
    thres = thres_ls[f1_ls.index(max(f1_ls))]
    return thres
