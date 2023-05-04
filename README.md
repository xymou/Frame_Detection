# Frame_Detection
This repository is the implementation of our COLING 22 paper [A Two Stage Adaptation Framework for Frame Detection via Prompt Learning]().
Our implementation is based on the project [OpenPrompt](https://github.com/thunlp/OpenPrompt), many thanks to THU for the open-source resource.

## Dataset
we get datasets available mainly from three resources: (1) news media (articles), (2) social media, (3) debates and statements. Five datasets are listed.  
- [mfc](https://aclanthology.org/P15-2072.pdf)
- [gvfc](https://aclanthology.org/K19-1047.pdf)
- [twitter](https://aclanthology.org/P17-1069.pdf)
- [immi](https://aclanthology.org/2021.naacl-main.179.pdf)
- [fora](https://aclanthology.org/N19-1142.pdf)

We use mfc for pre-training, and use the others for downstream generalization.

## Pre-training
see pre-train directory.

## Generalization
Initialize the components with pre-trained parameters and tune on downstream tasks.  
see gene directory.  
Our pre-trained model can be download through [link](https://drive.google.com/file/d/1zDxZfSoKRkikKWZi_j_Vf7dMaPvVQNec/view?usp=sharing)
