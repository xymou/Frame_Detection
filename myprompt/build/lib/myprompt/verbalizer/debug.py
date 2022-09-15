import torch
import torch.nn as nn
from adaptive_verbalizer import AdpativeVerbalizer, MultiAnswerAdaptiveVerbalizer
from hierarchy_verbalizer import WeightedVerbalizer
from myprompt.plm import load_plm
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel

plm, tokenizer, model_config, WrapperClass = load_plm('bert', '/remote-home/xymou/bert/bert-base-uncased/')
model = BertModel.from_pretrained('/remote-home/xymou/bert/bert-base-uncased/')
parent_classes = {
    'tax':['Economic'],
    'salary':['Economic'],
    'human':['Morality']
}

parent_concepts= {
    "Economic":['economic','benefit','cost'], 
    "Capacity and Resources":['resources'], 
    "Morality":['morality','ethics'], 
    "Fairness and Equality":['fairness'], 
    "Legality, Constitutionality, Jurisdiction":['law'], 
    "Policy Prescription and Evaluation":['policy'],
    "Crime and Punishment":['crime'], 
    "Security and Defense":['security'], 
    "Health and Safety":['health','safety'],
    "Quality of Life":['quality'], 
    "Cultural Identity":['culture'], 
    "Public Sentiment":['public'], 
    "Political":['partisan'],
    "External Regulation and Reputation":['u.s.']
}
num_classes = 3
concepts= {
    'tax':['economic','benefit','cost'],
    'salary':['economic','benefit','cost'],
    'human':['morality','ethics']

}

myverbalizer = WeightedVerbalizer(
    tokenizer =tokenizer,
    classes =list(concepts.keys()),
    concepts = concepts

)



# num_classes = 14
# label_text = {
#         "Economic":['cost','benefit','monetary','financial','trade','market','wage','employment',
#                     'unemployment','tax','economic consequence'], 
#         "Capacity and Resources":['lack of time','lack of space','lack of resources','capacity'],
#         "Morality":['religious','interpretation','duty','honor','righteousness','ethics',
#                               'responsibility','morality'],
#         "Fairness and Equality":['equality','fairness','right','discrimination'],
#         "Legality, Constitutionality, Jurisdiction":['legal','constitutional','jurisdictional','law',
#                                     'revision'],
#         "Policy Prescription and Evaluation":['policy evaluation','influence','effect'],
#         "Crime and Punishment":['crime','punishment','offence','illegal'],
#         "Security and Defense":['threat','security','border','defense'],
#         "Health and Safety":['health','safety','illness','disease','sanitation','carnage',
#                             'obesity','infrastructure'],
#         "Quality of Life":['quality of life','convenience','wealth','mobility','access','happiness',
#                           'ease'],
#         "Cultural Identity":['norms','trends','customs','culture','anniversary'],    
#         "Public Sentiment":['attitudes','opinion','protest','polling',
#                            'demographic','walkout'],  
#         "Political":['partisan','bipartisan','deal-making','vote','republican',
#                                              'democratic','congress'], 
#         "External Regulation and Reputation":['external relation','comparison','external reputation']

# }

# myverbalizer = MultiAnswerAdaptiveVerbalizer(
#     tokenizer =tokenizer,
#     plm=plm,
#     classes =classes,
#     label_text=label_text,
#     init_label_text=True,
#     label_text_sentence=False,
#     sent_encoder= model
# )

