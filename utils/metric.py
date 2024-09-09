import numpy as np
import torch
from datasets import load_metric
# !python3 -m pip install rouge_score

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

def RDASS(doc, target, pred): # (원문, 요약 target, 요약 pred)*B
  # 한 문장씩 들어옴. 모두 text가 아닌 semantic vector
  def cos_sim(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
  return (cos_sim(doc, pred)+cos_sim(target, pred)) / 2
  
def ROUGE(pred, target):
  rouge_score = load_metric("rouge")
  scores = rouge_score.compute(
    predictions=pred, references=target
  )
  return scores
  
