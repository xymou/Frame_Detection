"""
Adaptive verbalizer initialized by relation label word/ label description
reference paper: knowprompt: knowledge-aware prompt-tuning with synergistic optimization for relation extraction
"""
import json
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from yacs.config import CfgNode
from myprompt.data.example import InputFeatures
import re
from myprompt.verbalizer.base import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from myprompt.utils.logging import logger
from torch.nn.utils.rnn import pad_sequence
from transformers.file_utils import ModelOutput



"""
1个label对应一个虚拟答案
"""
class AdpativeVerbalizer(nn.Module):
    r"""
    TODO
    label_text: phrases的list或者对答案的描述文本
    e.g., {'School Safety':['safety in school','school safety'...],...}
    e.g., {'School Safety':['Stories about school safety or school violence']}
    对隐式答案的初始化，实现两种方式: word embedding的平均 / 直接使用BERT句向量的平均
    原文的relation type和embedding是一对一关系，但是我们可能提供多段描述/短语，
    因此还有一个加权或者聚合的过程(这个聚合可以在形成抽象答案之前做，目前是在初始化表示时直接平均)，或者形成多个答案，再agg
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 plm: PreTrainedModel,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_text: Optional[Mapping[str, str]] = None,
                 init_label_text: Optional[bool] = True,
                 label_text_sentence: Optional[bool] =False, #是否使用句向量
                 sent_encoder: Optional[PreTrainedModel] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True
                ):
        super().__init__()
        self.tokenizer = tokenizer
        self.plm = plm
        self.classes = classes
        if classes is not None and num_classes is not None:
            assert len(classes) == num_classes, "len(classes) != num_classes, Check you config."
            self.num_classes = num_classes
        elif num_classes is not None:
            self.num_classes = num_classes
        elif classes is not None:
            self.num_classes = len(classes)
        else:
            self.num_classes = None        
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_text = label_text
        self.init_label_text = init_label_text
        self.sent_encoder = sent_encoder
        self.post_log_softmax = post_log_softmax
        if self.init_label_text:
            if label_text_sentence:
                self._init_label_word_by_sent()
            else:
                self._init_label_word_by_word()
        else:
            self._init_label_word_random()


    @property
    def label_text(self,):
        r'''
        Label words means the words in the vocabulary projected by the labels. 
        E.g. if we want to establish a projection in sentiment classification: positive :math:`\rightarrow` {`wonderful`, `good`},
        in this case, `wonderful` and `good` are label words.
        '''
        return self._label_text       

    @label_text.setter
    def label_text(self, label_text):
        if label_text is None:
            return
        self._label_text = self._match_label_words_to_label_ids(label_text)


    def _match_label_words_to_label_ids(self, label_words):
        """
        sort label words dict of verbalizer to match the label order of the classes
        """
        if isinstance(label_words, dict):
            if self.classes is None:
                raise ValueError("""
                classes attribute of the Verbalizer should be set since your given label words is a dict.
                Since we will match the label word with respect to class A, to A's index in classes
                """)
            if set(label_words.keys()) != set(self.classes):
                raise ValueError("name of classes in verbalizer are differnt from those of dataset")
            label_words = [ # sort the dict to match dataset
                label_words[c]
                for c in self.classes
            ] # length: label_size of the whole task
        elif isinstance(label_words, list) or isinstance(label_words, tuple):
            pass
            # logger.info("""
            # Your given label words is a list, by default, the ith label word in the list will match class i of the dataset.
            # Please make sure that they have the same order.
            # Or you can pass label words as a dict, mapping from class names to label words.
            # """)
        else:
            raise ValueError("Verbalizer label words must be list, tuple or dict")
        return label_words        


    def get_label_word(self,):
        #tokenizer the label text
        ans=[]
        for label in self.label_text:
            tmp=[]
            if isinstance(label,list):
                for text in label:
                    text = text.lower()
                    text_word_id = self.tokenizer(text, add_special_tokens= False)['input_ids']
                    print(text, text_word_id)
                    tmp.append(torch.tensor(text_word_id))
                padded_label_word_list = pad_sequence([x for x in tmp], batch_first=True, padding_value=0)
                ans.append(padded_label_word_list)
            else:
                raise ValueError('Please input description or phrases as list of text.')
        return ans

    def _init_label_word_by_word(self,):
        self.tokenizer.add_special_tokens({'additional_special_tokens':[f"[class{i}]" for i in range(len(self.classes))]})  
        self.plm.resize_token_embeddings(len(self.tokenizer))      
        label_words = self.get_label_word()
        num_labels = len(label_words)
        continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(num_labels)], add_special_tokens=False)['input_ids']]
        with torch.no_grad():
            word_embeddings = self.plm.get_input_embeddings()
            for j in range(len(label_words)):
                res=[]
                for i, idx in enumerate(label_words[j]):
                    res.append(torch.mean(word_embeddings.weight[idx], dim=0)) #每个token 取平均值得到这个短语的表示
                word_embeddings.weight[continous_label_word[j]] = torch.mean(torch.cat(res,dim =0), dim=0) #每个短语取平均值得到这个label word的表示
            assert torch.equal(self.plm.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.plm.get_input_embeddings().weight, self.plm.get_output_embeddings().weight)
        self.word2label = continous_label_word
        

    def _init_label_word_by_sent(self,):
        assert self.sent_encoder!=None
        self.tokenizer.add_special_tokens({'additional_special_tokens':[f"[class{i}]" for i in range(len(self.classes))]})   
        self.plm.resize_token_embeddings(len(self.tokenizer))      
        num_labels= len(self.classes)
        continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(num_labels)], add_special_tokens=False)['input_ids']]
        with torch.no_grad():
            word_embeddings = self.plm.get_input_embeddings()
            for j in range(len(self.label_text)):
                res=[]
                for i in range(len(self.label_text[j])):
                    inputs = torch.tensor(self.tokenizer(self.label_text[j][i])['input_ids']).view(1,-1)
                    res.append(self.sent_encoder(inputs)[1]) #pooler_ouput, 句向量
                word_embeddings.weight[continous_label_word[j]] = torch.mean(torch.cat(res,dim =0), dim=0) #每个句子取平均值得到这个label word的表示
            assert torch.equal(self.plm.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.plm.get_input_embeddings().weight, self.plm.get_output_embeddings().weight)
        self.word2label = continous_label_word

    def _init_label_word_random(self,):
        self.tokenizer.add_special_tokens({'additional_special_tokens':[f"[class{i}]" for i in range(len(self.classes))]})  
        self.plm.resize_token_embeddings(len(self.tokenizer))      
        num_labels = len(self.classes)
        continous_label_word = [a[0] for a in self.tokenizer([f"[class{i}]" for i in range(num_labels)], add_special_tokens=False)['input_ids']]
        with torch.no_grad():
            word_embeddings = self.plm.get_input_embeddings()
            for j in range(num_labels):
                word_embeddings.weight[continous_label_word[j]] = torch.rand_like(word_embeddings.weight[0])
            assert torch.equal(self.plm.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.plm.get_input_embeddings().weight, self.plm.get_output_embeddings().weight)
        self.word2label = continous_label_word

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        #batch_size * vocab_size
        label_words_logits = logits[:, self.word2label]
        return label_words_logits

    def process_outputs(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures],
                       **kwargs):

        return self.process_logits(outputs, batch=batch, **kwargs)

    def process_logits(self, logits: torch.Tensor, **kwargs):
        # project
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) 

        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        # aggreate 抽象答案和标签暂时是一对一的关系，不用agg
        # label_logits = self.aggregate(label_words_logits)
        return label_words_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:

        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)


    # def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
    #     raise NotImplementedError      

    def gather_outputs(self, outputs: ModelOutput):
        return outputs.logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() ==  1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,keepdim=True) # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

    


"""
一个label对应多个虚拟答案
"""
class MultiAnswerAdaptiveVerbalizer(nn.Module):
    r"""
    label_text: phrases的list或者对答案的描述文本
    e.g., {'School Safety':['safety in school','school safety'...],...}
    e.g., {'School Safety':['Stories about school safety or school violence']}
    对隐式答案的初始化，实现两种方式: word embedding的平均 / 直接使用BERT句向量的平均
    原文的relation type和embedding是一对一关系，但是我们可能提供多段描述/短语，
    形成多个答案, 和manual verbalizer的多个答案一样, 处理完再agg
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 plm: PreTrainedModel,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_text: Optional[Mapping[str, str]] = None,
                 init_label_text: Optional[bool] = True,
                 label_text_sentence: Optional[bool] =False, #是否使用句向量
                 sent_encoder: Optional[PreTrainedModel] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                 refinement: Optional[bool] = False, #whether to assign a learnable weight for each label word
                 label_avg: Optional[str] = True #use average probs of label word or not
                ):
        super().__init__()
        self.tokenizer = tokenizer
        self.plm = plm
        self.classes = classes
        if classes is not None and num_classes is not None:
            assert len(classes) == num_classes, "len(classes) != num_classes, Check you config."
            self.num_classes = num_classes
        elif num_classes is not None:
            self.num_classes = num_classes
        elif classes is not None:
            self.num_classes = len(classes)
        else:
            self.num_classes = None        
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_text = label_text
        self.init_label_text = init_label_text
        self.sent_encoder = sent_encoder
        self.post_log_softmax = post_log_softmax
        if self.init_label_text:
            if label_text_sentence:
                self._init_label_word_by_sent()
            else:
                self._init_label_word_by_word()
        else:
            self._init_label_word_random()
        self.generate_parameters()
        self.refinement = refinement
        self.label_avg = label_avg        
        if self.refinement:
            self.word_weight = nn.Parameter(torch.zeros_like(self.label_words_mask, dtype=torch.float32), requires_grad = True)
        else:
            assert self.refinement==False and self.label_avg == True

    @property
    def label_text(self,):
        r'''
        Label words means the words in the vocabulary projected by the labels. 
        E.g. if we want to establish a projection in sentiment classification: positive :math:`\rightarrow` {`wonderful`, `good`},
        in this case, `wonderful` and `good` are label words.
        '''
        return self._label_text       

    @label_text.setter
    def label_text(self, label_text):
        if label_text is None:
            return
        self._label_text = self._match_label_words_to_label_ids(label_text)


    def _match_label_words_to_label_ids(self, label_words):
        """
        sort label words dict of verbalizer to match the label order of the classes
        """
        if isinstance(label_words, dict):
            if self.classes is None:
                raise ValueError("""
                classes attribute of the Verbalizer should be set since your given label words is a dict.
                Since we will match the label word with respect to class A, to A's index in classes
                """)
            if set(label_words.keys()) != set(self.classes):
                raise ValueError("name of classes in verbalizer are differnt from those of dataset")
            label_words = [ # sort the dict to match dataset
                label_words[c]
                for c in self.classes
            ] # length: label_size of the whole task
        elif isinstance(label_words, list) or isinstance(label_words, tuple):
            pass
        else:
            raise ValueError("Verbalizer label words must be list, tuple or dict")
        return label_words        

    def get_label_word(self,):
        #tokenizer the label text
        ans=[]
        for label in self.label_text:
            tmp=[]
            if isinstance(label,list):
                for text in label:
                    text = text.lower()
                    text_word_id = self.tokenizer(text, add_special_tokens= False)['input_ids']
                    print(text, text_word_id)
                    tmp.append(torch.tensor(text_word_id))
                padded_label_word_list = pad_sequence([x for x in tmp], batch_first=True, padding_value=0)
                ans.append(padded_label_word_list)
            else:
                raise ValueError('Please input description or phrases as list of text.')
        return ans

    def _init_label_word_by_word(self,):     
        label_words = self.get_label_word()
        num_labels = len(label_words)
        special_tokens= []
        for i in range(num_labels):
            for j in range(len(label_words[i])):
                special_tokens.append(f"[class{i}-answer{j}")
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})  
        self.plm.resize_token_embeddings(len(self.tokenizer))         
        continous_label_word = []
        for i in range(num_labels):
            tmp =  [a[0] for a in self.tokenizer([f"[class{i}-answer{j}]" for j in range(len(label_words[i]))], add_special_tokens=False)['input_ids']]
            continous_label_word.append(tmp)
        with torch.no_grad():
            word_embeddings = self.plm.get_input_embeddings()
            for i in range(len(label_words)):
                for j, idx in enumerate(label_words[i]):
                    word_embeddings.weight[continous_label_word[i][j]] = torch.mean(word_embeddings.weight[idx], dim=0)
            assert torch.equal(self.plm.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.plm.get_input_embeddings().weight, self.plm.get_output_embeddings().weight)
        self.word2label = continous_label_word      

    def _init_label_word_by_sent(self,):
        assert self.sent_encoder!=None
        num_labels= len(self.classes)
        label_words = self.label_text
        special_tokens= []
        for i in range(num_labels):
            for j in range(len(label_words[i])):
                special_tokens.append(f"[class{i}-answer{j}")
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})   
        self.plm.resize_token_embeddings(len(self.tokenizer))      
        continous_label_word = []
        for i in range(num_labels):
            tmp =  [a[0] for a in self.tokenizer([f"[class{i}-answer{j}]" for j in range(len(label_words[i]))], add_special_tokens=False)['input_ids']]
            continous_label_word.append(tmp)
        with torch.no_grad():
            word_embeddings = self.plm.get_input_embeddings()
            for i in range(len(self.label_text)):
                for j in range(len(self.label_text[i])):
                    inputs = torch.tensor(self.tokenizer(self.label_text[i][j])['input_ids']).view(1,-1)
                    res = self.sent_encoder(inputs)[1] #pooler_ouput, 句向量
                    word_embeddings.weight[continous_label_word[i][j]] = res
                    print(continous_label_word[i][j], word_embeddings.weight[continous_label_word[i][j]].size())
            assert torch.equal(self.plm.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.plm.get_input_embeddings().weight, self.plm.get_output_embeddings().weight)
        self.word2label = continous_label_word
    
    def _init_label_word_random(self):
        label_words = self.get_label_word()
        num_labels = len(label_words)
        special_tokens= []
        for i in range(num_labels):
            for j in range(len(label_words[i])):
                special_tokens.append(f"[class{i}-answer{j}")
        self.tokenizer.add_special_tokens({'additional_special_tokens':special_tokens})  
        self.plm.resize_token_embeddings(len(self.tokenizer))         
        continous_label_word = []
        for i in range(num_labels):
            tmp =  [a[0] for a in self.tokenizer([f"[class{i}-answer{j}]" for j in range(len(label_words[i]))], add_special_tokens=False)['input_ids']]
            continous_label_word.append(tmp)
        with torch.no_grad():
            word_embeddings = self.plm.get_input_embeddings()
            for i in range(len(label_words)):
                for j, idx in enumerate(label_words[i]):
                    word_embeddings.weight[continous_label_word[i][j]] = torch.rand_like(word_embeddings.weight[0])
            assert torch.equal(self.plm.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.plm.get_input_embeddings().weight, self.plm.get_output_embeddings().weight)
        self.word2label = continous_label_word         


    def generate_parameters(self):
        all_ids = self.word2label
        max_len = 1
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = [[[1] for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        words_ids = [[[ids] for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]             
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:

        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000*(1-self.label_words_mask)
        return label_words_logits

    def process_outputs(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures],
                       **kwargs):

        return self.process_logits(outputs, batch=batch, **kwargs)


    def process_logits(self, logits: torch.Tensor, **kwargs):
        # project
        label_words_logits = self.project(logits, **kwargs)  #Output: (batch_size, num_classes) or  (batch_size, num_classes, num_label_words_per_label)
        
        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        # aggreate
        label_logits = self.aggregate(label_words_logits)
        return label_logits        

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:

        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)


    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        if self.label_avg:
            label_words_logits = (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1)
        else:
            weight = F.softmax(self.word_weight-10000*(1-self.label_words_mask), dim = 1)
            weight = torch.stack([weight] * label_words_logits.size(0), dim=0)
            label_words_logits = (label_words_logits * self.label_words_mask * weight).sum(-1)
        return label_words_logits

    def gather_outputs(self, outputs: ModelOutput):
        return outputs.logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() ==  1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,keepdim=True) # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

    def handle_multi_token(self, label_words_logits, mask):
        if self.multi_token_handler == "first":
            label_words_logits = label_words_logits.select(dim=-1, index=0)
        elif self.multi_token_handler == "max":
            label_words_logits = label_words_logits - 1000*(1-mask.unsqueeze(0))
            label_words_logits = label_words_logits.max(dim=-1).values
        elif self.multi_token_handler == "mean":
            label_words_logits = (label_words_logits*mask.unsqueeze(0)).sum(dim=-1)/(mask.unsqueeze(0).sum(dim=-1)+1e-15)
        else:
            raise ValueError("multi_token_handler {} not configured".format(self.multi_token_handler))
        return label_words_logits