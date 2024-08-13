# How Do Moral Emotions Shape Political Participation? A Cross-Cultural Analysis of Online Petitions Using Language Models

This repository contains data and model for the ACL 2024 paper, [How Do Moral Emotions Shape Political Participation? A Cross-Cultural Analysis of Online Petitions Using Language Models](https://aclanthology.org/2024.findings-acl.963/).

## Abstract
Understanding the interplay between emotions in language and user behaviors is critical. We study how moral emotions shape political participation of users based on cross-cultural online petition data. To quantify moral emotions, we employ a context-aware NLP model that is designed to capture the subtle nuances of emotions across cultures. For model training, we construct and share a moral emotion dataset comprising 50,000 petition sentences in Korean and English along with emotion labels annotated by a fine-tuned LLM. We examine two distinct types of user participation: general support (i.e., registered signatures of petitions) and active support (i.e., sharing petitions on social media). We discover that moral emotions like other-suffering increase both forms of participation and help petitions go viral, while self-conscious have the opposite effect. The most prominent moral emotion, other-condemning, led to polarizing responses among the audience. In contrast, other-praising was perceived differently by culture; it led to a rise in active support in Korea but a decline in the UK. Our findings suggest that both moral emotions embedded in language and cultural perceptions are critical in engaging the public in political discourse.

## Method Overview
![Overview](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/image/overview.png)
Method overview. We propose a full framework for constructing data, modeling classification, and conducting analysis on the theme of moral emotions.

## Result
![Result](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/image/result.png)
Predictive margins for general support (depicted by a red line) and active support (blue line) across five emotions in studied countries, with 95% confidence intervals.

## Data
We collected data from government-led petition platforms from [the Korean government archive](http://webarchives.pa.go.kr/19th/www.president.go.kr/petitions/) and [the UK Government and Parliament Petition website](https://petition.parliament.uk/).

### 1. Human Annotation Dataset
- [Korean](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/data/KOR_Human_Annotation_Dataset.parquet)
- [English](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/data/ENG_Human_Annotation_Dataset.parquet)
- Columns: Question, User1, User2, User3, User4, User5
- NA data is a response of 'Hard to tell'

### < Human Annotation Dataset Sample >
|   | Question | User1 | User2 | User3 | User4 | User5 |
|---|----------|-------|-------|-------|-------|-------|
| 0 | Use the minstrel code to have them removed from office! | Other-condemning | Other-condemning | Other-condemning | Other-condemning | Other-condemning |
| 1 | Martin Lewis has made a fantastic contribution to educating the people of Britain on how to manage their finances. | Other-praising | Other-praising | Other-praising | Other-praising | Other-praising |
| 2 | The UK government should actively respond and allow Ukrainian refugees into the country and potentially even facilitate transport of the vulnerable, elderly and children, into the country. | Other-suffering | Other-suffering | Other-suffering | Other-suffering | Other-suffering |
| 3 | Because today I feel embarrassed to be a nurse. | Self-conscious | Self-conscious | Self-conscious | Self-conscious | Self-conscious |
| 4 | Instead of going dark at 5.15pm it would be 6.15pm. | Neutral | Neutral | Neutral | Neutral | Neutral |
| 5 | I am afraid that I will catch coronavirus from one of them. | Non-moral-emotion | Non-moral-emotion | Non-moral-emotion | Non-moral-emotion | Non-moral-emotion |

### 2. Moral Emotion Dataset
- [Korean](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/data/KOR_Moral_Emotion_Dataset.parquet) (49,930)
- [English](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/data/ENG_Moral_Emotion_Dataset.parquet) (49,896)
- Columns: Question, Label, Label_Type

### < Moral Emotion Dataset Sample >
|   | Text | Label | Label_Type |
|---|----------------------------------------------------------------------------------------|------------------------------------|------------|
| 0 | Because it’s amazing what they are doing words can’t describe what a fantastic job they are doing and they deserve all the recognition they’re getting and I know you agree with me! | Other-praising | Single |
| 1 | Stop putting us victims in danger.Abusers work way to quickly and smartly now and there’s too many ways they get around the current law as it stands.Make restraining orders and domestic sentences public knowledge. | Other-condemning, Other-suffering | Multi |
| 2 | Trafalgar Square has always been where Britain's war heroes are honoured with statues. | Neutral | Single |
| 3 | Many health care workers are working without adequate PPE due to underfunding of the NHS making them ill-equipped to handle the COVID-19 crisis. | Other-condemning, Other-suffering | Multi |
  
## Model
```python
# Detail Example: https://github.com/Paul-scpark/Moral-Emotion/blob/main/model_inference.py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def inference_single_model(df, model_name):
    pipe = pipeline("text-classification", model=model_name, top_k=None)
    results = pipe(df['Question'].tolist())
    output_df = pd.DataFrame([{item['label']: item['score'] for item in row} for row in results])
    
    return output_df

''' 'output_df' shape
| Non-Moral-Emotion |  Neutral  | Other-Suffering | Other-Praising | Other-Condemning | Self-Conscious |
|-------------------|-----------|-----------------|----------------|------------------|----------------|
| 0.621278          | 0.389092  | 0.026186        | 0.022388       | 0.018874         | 0.004896       |
| 0.612030          | 0.390416  | 0.027614        | 0.022423       | 0.019170         | 0.004669       |
| 0.554191          | 0.386037  | 0.036295        | 0.021075       | 0.022980         | 0.003603       |
| 0.681116          | 0.312530  | 0.023661        | 0.017224       | 0.020274         | 0.004694       |
| 0.547903          | 0.446682  | 0.030198        | 0.020471       | 0.018342         | 0.004556       |
'''
```
- Korean
  - [BERT](https://huggingface.co/Chaeyoon/BERT-Moral-Emotion-KOR)
  - [RoBERTa](https://huggingface.co/Chaeyoon/RoBERTa-Moral-Emotion-KOR)
  - [ELECTRA](https://huggingface.co/Chaeyoon/ELECTRA-Moral-Emotion-KOR)
- English
  - [BERT](https://huggingface.co/Chaeyoon/BERT-Moral-Emotion-ENG)
  - [RoBERTa](https://huggingface.co/Chaeyoon/RoBERTa-Moral-Emotion-ENG)
  - [ELECTRA](https://huggingface.co/Chaeyoon/ELECTRA-Moral-Emotion-ENG)

## Citation
```
@inproceedings{
  kim-etal-2024-moral-emotions,
  title={How {D}o {M}oral {E}motions {S}hape {P}olitical {P}articipation? {A} {C}ross-{C}ultural {A}nalysis of {O}nline {P}etitions {U}sing {L}anguage {M}odels},
  author={Kim, Jaehong and Jeong, Chaeyoon and Park, Seongchan and Cha, Meeyoung and Lee, Wonjae},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  year={2024},
  url={https://aclanthology.org/2024.findings-acl.963},
  pages={16274--16289}
}
```
