from myprompt.utils.logging import logger
from myprompt.data.example import InputExample, InputFeatures
from typing import *
from transformers.tokenization_utils import PreTrainedTokenizer
from myprompt.template.base import Template

class ManualTemplate(Template):
    """
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        mask_token (:obj:`str`, optional): The special token that is masked and need to be predicted by the model. Default to ``<mask>``
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """
    registered_inputflag_names = ["loss_ids", 'shortenable_ids']
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[List[str]] = None,
                 mask_token: str = '<mask>',
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer, 
                         mask_token=mask_token,
                         placeholder_mapping=placeholder_mapping)
        self.text = text
    
    def on_text_set(self):
        """
        when template text was set
        
        1. parse text
        """

        self.text = self.parse_text(self.text)        