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


class LabelEmbedVerbalizer(nn.Module):
    r"""
    TODO
    label_text: phrases的list或者对答案的描述文本
    e.g., {'School Safety':['safety in school','school safety'...],...}
    e.g., {'School Safety':['Stories about school safety or school violence']}
    """
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 plm: PreTrainedModel,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_text: Optional[Mapping[str, str]] = None,
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
        self.post_log_softmax = post_log_softmax


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
                    #print(text, text_word_id)
                    tmp.append(torch.tensor(text_word_id))
                padded_label_word_list = pad_sequence([x for x in tmp], batch_first=True, padding_value=0)
                ans.append(padded_label_word_list)
            else:
                raise ValueError('Please input description or phrases as list of text.')
        return ans

    def get_label_embed(self,):
   
        label_words = self.get_label_word()
        num_labels = len(label_words)
        with torch.no_grad():
            embeddings = []
            word_embeddings = self.plm.get_input_embeddings()
            for j in range(len(label_words)):
                res=[]
                for i, idx in enumerate(label_words[j]):
                    res.append(torch.mean(word_embeddings.weight[idx], dim=0).view(1,-1)) 
                embeddings.append(torch.mean(torch.cat(res,dim =0), dim=0).view(1,-1))     
            self.label_embed = torch.cat(embeddings,dim=0)
        

    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        #batch_size * vocab_size
        raise NotImplementedError

    def process_outputs(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures],
                       **kwargs):

        return self.process_logits(outputs, batch=batch, **kwargs)

    def process_logits(self, mlm_logits: torch.Tensor, **kwargs):
        
        self.get_label_embed()
        logits = torch.mm(mlm_logits, self.label_embed.T)
        logits = F.sigmoid(logits)
        return logits

    def process_logits2(self, mlm_logits: torch.Tensor, **kwargs):
        # 不加sigmoid
        self.get_label_embed()
        logits = torch.mm(mlm_logits, self.label_embed.T)
        return logits    

    def gather_outputs(self, outputs: ModelOutput):
        return outputs
