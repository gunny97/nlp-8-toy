import gc
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import sys
from model.fine_tune_cls import TransformerModule
from utils.loader.cls_datamodule import ClsDataModule
from utils.config.cls_config import add_options
import numpy as np
import pandas as pd
from utils.metric import F1
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def get_model(config) -> TransformerModule:
    
    if config.target_name == 'topic':
        num_classes = 9
        
    elif config.target_name == 'keyword':
        num_classes = 87
        
    elif config.target_name == 'speech_act':
        num_classes = 3
        
    else:
        raise ValueError(f"invalid target_name : {config.target_name}")
    
    
    if config.target_name in ['topic', 'keyword']:
        dataset_size = len(pd.read_csv(config.val_data))
    elif config.target_name == 'speech_act':
        dataset_size = len(pd.read_csv(config.val_act_data))
    model = TransformerModule(
        pretrained_model=config.pretrained_model,
        num_classes=num_classes,
        lr=config.lr,
        use_quantization=config.use_quantization,
        dataset_size=dataset_size,
        epoch=config.epochs,
        batch_size=config.batch_size,
        warmup=int(config.warmup_ratio * dataset_size * config.epochs / config.batch_size),
        checkpoint_path = config.checkpoint_path
    )

    dm = ClsDataModule(
        train_data=config.train_data,
        val_data=config.val_data,
        train_act_data=config.train_act_data,
        val_act_data=config.val_act_data,
        target_name=config.target_name,
        pretrained_model=config.pretrained_model,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    dm.setup('test')
    return model.cuda(), dm.test_dataloader()

'''def draw_ROC(preds, targets, config):
    fpr, tpr, thresholds = roc_curve(np.array(preds), np.array(targets))
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC 곡선 (면적 = {:.2f})'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.grid()
    filename = f"./cls/utils/eval/ROC/{config.pretrained_model[:-5]}_ROC"
    plt.savefig(filename, dpi=200)
    plt.close()'''
    
def evaluate_cls(model, dataloader, config):
    model.eval()
    torch.manual_seed(0)
    if config.target_name == 'topic':
        num_classes = 9
        
    elif config.target_name == 'keyword':
        num_classes = 87
        
    elif config.target_name == 'speech_act':
        num_classes = 3
        
    else:
        raise ValueError(f"invalid target_name : {config.target_name}")
    
    preds = []
    targets = []
    # Evaluation
    for batch in dataloader:
        # Get Data
        with torch.no_grad():
            # Forward
            outputs = model(input_ids=batch['input_ids'].cuda(),
                            attention_mask=batch['attention_mask'].cuda(),
                            labels=batch['labels'].cuda())
                           
            # (B, 1)
            pred_class = torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1)
            preds.extend(pred_class.tolist())
            targets.extend(batch['labels'].tolist())
    # Calculate Score
    accuracy = np.sum(np.array([1 if a==b else 0 for a, b, in zip(preds, targets)]))/len(preds)
    f1_score = F1(preds, targets, num_classes)
    # ROC-curve는 이진 분류에서만 작동하고 다중 분류도 할 수 있는데 느려질까봐, 볼 ROC-curve가 너무 많을까봐 보류
    #draw_ROC(preds, targets, config) 
    return accuracy, f1_score

if __name__ == "__main__":
    # Free up gpu vRAM from memory leaks.
    torch.cuda.empty_cache()
    gc.collect()
        
    train_config = add_options()

    # Get model
    model, dataloader = get_model(train_config)
    accuracy, f1_score = evaluate_cls(model, dataloader, train_config)
    print('accuracy : ', accuracy)
    print('f1-score : ', f1_score)