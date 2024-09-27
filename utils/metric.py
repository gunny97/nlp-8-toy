import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer

# !python3 -m pip install rouge_score

def get_embedding(text, model, tokenizer, device='cuda'):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰에 대한 임베딩을 사용하거나, 문장 평균을 취할 수 있음
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# classification
def get_precision(TP, FP):
  return np.where(TP+FP > 0, TP/(TP+FP), 0)
      
def get_recall(TP, FN):
  return np.where(TP+FN > 0, TP/(TP+FN), 0)
    
def f1(outputs, targets, num_class):
  # outputs, targets -> (B,)
  TP, FP, FN = np.zeros(num_class, dtype=int), np.zeros(num_class, dtype=int), np.zeros(num_class, dtype=int)
  for pred, targ in zip(outputs, targets):
    if pred==targ:  # TP
      TP[targ] += 1
    else:
      FP[pred] += 1
      FN[targ] += 1
      
  precision, recall = get_precision(TP, FP), get_recall(TP, FN)
  score = np.where(precision+recall > 0, 2*precision*recall / (precision+recall), 0)
  return np.mean(score)

# generation

def RDASS(doc, target, pred, model, tokenizer, device='cuda'): # (원문, 요약 target, 요약 pred)*B
  # 한 문장씩 들어옴. 모두 text가 아닌 semantic vector
  def cos_sim(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    
  # 임베딩 벡터 계산
  doc_emb = get_embedding(doc, model, tokenizer, device)
  target_emb = get_embedding(target, model, tokenizer, device)
  pred_emb = get_embedding(pred, model, tokenizer, device)
    
  return (cos_sim(doc_emb, pred_emb) + cos_sim(target_emb, pred_emb)) / 2
  
def ROUGE_N(pred, target, n): # (요약 pred, 요약 target, n-gram size)
  scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True) # Rouge-N 점수를 계산할 RougeScorer
  scores = scorer.score(target, pred) # 예측 텍스트와 참조 텍스트에 대해 Rouge 점수 계산
  return scores[f'rouge{n}'] # recall, precision, fmeasure(F1 Score) 반환

def ROUGE_L(pred, target):
  scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
  scores = scorer.score(target, pred)
  return scores['rougeL']
    
