# How Do Moral Emotions Shape Political Participation? A Cross-Cultural Analysis of Online Petitions Using Language Models

This repository contains data for the ACL 2024 paper, [How Do Moral Emotions Shape Political Participation? A Cross-Cultural Analysis of Online Petitions Using Language Models](https://openreview.net/pdf?id=b3AoAk60mL).

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
- [Korean](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/data/KOR_Human_Annotation_Dataset.parquet) (Train: 300 / Test: 340)
- [English](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/data/ENG_Human_Annotation_Dataset.parquet) (Train: 300 / Test: 340)
- Columns: Question, Label, Label_Type, Dataset

### 2. Moral Emotion Dataset
- [Korean](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/data/KOR_Moral_Emotion_Dataset.parquet) (49,930)
- [English](https://github.com/Paul-scpark/Moral-Emotions-Political-Participation/blob/main/data/ENG_Moral_Emotion_Dataset.parquet) (49,896)
- Columns: Question, Label, Label_Type

### < Human Annotation / Moral Emotion Dataset Sample >
|   | Question | Label | Label_Type | Dataset |
|---|----------------------------------------------------------------------------------------|------------------------------------|------------|---------|
| 0 | Because it’s amazing what they are doing words can’t describe what a fantastic job they are doing and they deserve all the recognition they’re getting and I know you agree with me! | Other-Praising | Single | Train |
| 1 | Stop putting us victims in danger.Abusers work way to quickly and smartly now and there’s too many ways they get around the current law as it stands.Make restraining orders and domestic sentences public knowledge. | Other-Condemning, Other-Suffering | Multi | Train |
| 2 | Trafalgar Square has always been where Britain's war heroes are honoured with statues. | Neutral | Single | Test |
| 3 | Many health care workers are working without adequate PPE due to underfunding of the NHS making them ill-equipped to handle the COVID-19 crisis. | Other-Condemning, Other-Suffering | Multi | Test |
  
## Setup

## Citation
```
@inproceedings{
  kim-etal-2024-moral-emotions,
  title={How {D}o {M}oral {E}motions {S}hape {P}olitical {P}articipation? {A} {C}ross-{C}ultural {A}nalysis of {O}nline {P}etitions {U}sing {L}anguage {M}odels},
  author={Kim, Jaehong and Jeong, Chaeyoon and Park, Seongchan and Cha, Meeyoung and Lee, Wonjae},
  booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
  year={2024}
}
```
