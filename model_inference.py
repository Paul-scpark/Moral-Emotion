import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_names_kor = [
    "Chaeyoon/RoBERTa-Moral-Emotion-KOR",
    "Chaeyoon/ELECTRA-Moral-Emotion-KOR",
    "Chaeyoon/BERT-Moral-Emotion-KOR"
]

model_names_eng = [
    "Chaeyoon/RoBERTa-Moral-Emotion-ENG",
    "Chaeyoon/ELECTRA-Moral-Emotion-ENG",
    "Chaeyoon/BERT-Moral-Emotion-ENG"
]

def inference_single_model(df, model_name):
    pipe = pipeline("text-classification", model=model_name, top_k=None)
    results = pipe(df['Question'].tolist())
    output_df = pd.DataFrame([{item['label']: item['score'] for item in row} for row in results])
    
    return output_df

def inference_all_model(df, model_names):
    output_dic = {}
    for model_name in model_names:
        pipe = pipeline("text-classification", model=model_name, top_k=None)
        results = pipe(df['Question'].tolist())
        output_dic[model_name] = pd.DataFrame([{item['label']: item['score'] for item in row} for row in results])
        
    return output_dic

if __name__ == "main":
    ### KOR
    test_kor = pd.read_parquet("data/KOR_Human_Annotation_Dataset.parquet").reset_index(drop=True)
    kor_output_dic = inference_all_model(test_kor, model_names_kor)
    
    ### ENG
    test_eng = pd.read_parquet("data/ENG_Human_Annotation_Dataset.parquet").reset_index(drop=True)
    eng_output_dic = inference_all_model(test_eng, model_names_eng)