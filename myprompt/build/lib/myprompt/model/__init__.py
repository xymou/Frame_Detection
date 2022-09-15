from .base import PromptModel
from .classification import PromptForClassification
from .generation import PromptForGeneration
from .transfer_clf import PromptForTransferClassification
from .contrastive import PromptForContrastive
from .dual_template import PromptSharedModel,PromptForSharedTemplate, PromptForMergedGoal, PromptForDANN, PromptDiffDANNModel, PromptForDiffDANN
from .label_embedding import PromptForLabelMatching, PromptwoVerbalizer, PromptForSharedMLMTemplate, PromptSharedMLMModel,\
     PromptForSharedMLMTemplateMLP, PromptForNPNet, PromptIssueMLMModel, PromptForJointNPNet, PromptJointIssueMLMModel, PromptJointModel,\
          PromptForJointModel, PromptForShare
from .ptuning_v2 import BertPrefixForSequenceClassification, RobertaPrefixForSequenceClassification
from .prefix_share import PrefixShareMLM, PrefixShareAdvMLM, PrefixShareAdvMLM2, PrefixShareAdvMLM3
from .jointmodel import JointMLM, JointMLMAdvMlP, JointMLMAdvMlPLabelEmb, JointMLMDiff, JointMLMDiffMLP, JointMLMDiffMLPLabelEmb