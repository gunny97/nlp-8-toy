import os
import gc
import torch
import random
import argparse
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from model.fine_tune_sts import TransformerModule
from utils.data_module import STSDataModule


# seed 고정
torch.manual_seed(104)
torch.cuda.manual_seed(104)
torch.cuda.manual_seed_all(104)
random.seed(104)


def inference_loop(config) -> TransformerModule:
    model = TransformerModule(
        model_name=config.model_name,
        lr=config.lr,
        dataset_size=config.dataset_size,
        max_epoch=config.max_epoch,
        batch_size=config.batch_size,
        warmup=config.warmup,
        beta1=config.beta1,
        beta2=config.beta2,
        weight_decay=config.weight_decay,
        CL=config.CL,
    )

    dm = STSDataModule(
        model_name=config.model_name,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_workers=config.num_workers,
        train_path=config.predict_path,
        dev_path=config.predict_path,
        test_path=config.predict_path,
        predict_path=config.predict_path
    )

    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch,log_every_n_steps=1)
        
    predictions = trainer.predict(model=model, datamodule=dm, ckpt_path = args.ckpt_path)
    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    
    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./resources/sts/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('submission_0923_3.csv', index=False)
    

if __name__ == "__main__":
    # Free up gpu vRAM from memory leaks.
    torch.cuda.empty_cache()
    gc.collect()

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deliciouscat/kf-deberta-base-cross-sts', type=str, choices=['deliciouscat/kf-deberta-base-cross-sts'])
    parser.add_argument('--predict_path', default='./resources/sts/test.csv')
    parser.add_argument('--CL', default="code/code/augmentation_noswap_all_swap_CL/1500_0.9_0.999_0.01_epoch=9-val_loss=0.020.ckpt", help='Load Unsupervised Contrastive Learning Model')
    parser.add_argument('--ckpt_path', default='./code/code/augmentation_noswap_all_swap/1000_0.9_0.999_0.01_epoch=12-val_pearson=0.929.ckpt')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=20, type=int) 
    parser.add_argument('--dataset_size', default=10000, type=int) 
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps", choices=[500, 600, 1000, 2000])
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)

    args = parser.parse_args()
    
    # Train model.
    inference_loop(args)
