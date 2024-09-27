import gc
import os
import argparse
import numpy as np
import torch
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer
from utils.metric import RDASS, ROUGE_N, ROUGE_L
from model.fine_tune_gen import TransformerModuleForGen
from utils.loader.gen_datamodule import GenDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from utils.config.gen_config import add_options

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./resources/log/gen/epoch=24-Val_Loss=2.35.ckpt', type=str)
    args = parser.parse_args(args=[])
    config = add_options()

    # 데이터 모듈 초기화
    dm = GenDataModule(
        train_data=config.train_data,
        valid_data=config.test_data,
        pretrained_model=config.pretrained_model,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # 트레이너 초기화
    tainer = Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=1,
        accumulate_grad_batches=4,
    )

    # 모델 로드
    model = TransformerModuleForGen.load_from_checkpoint(
        args.model_path,
        pretrained_model=config.pretrained_model,
        lr=config.lr,
        max_length=config.max_length,
        strict=False
    )
    model.eval()

    dm.setup(stage='test')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    embedding_model = AutoModel.from_pretrained(config.pretrained_model).to(device)

    test_loader = dm.test_dataloader()
    for batch in test_loader:
        inputs = batch['input_ids']
        inputs_ = tokenizer(
            batch["input"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        targets = batch['label']

        decoded_input = tokenizer.decode(inputs_['input_ids'][0], skip_special_tokens=True)
        print(decoded_input)
        
        predictions = model.model.generate(**inputs_, max_new_tokens=64, use_cache=True)
        predictions = [
            tokenizer.decode(pred)
            .split("<start_of_turn>model")[-1]
            .split("<end_of_turn>")[0]
            .strip()
            for pred in predictions
        ]

        documents = [model.tokenizer.decode(input_ids.tolist(), skip_special_tokens=True) for input_ids in inputs]
        predictions = [model.tokenizer.decode(pred.tolist(), skip_special_tokens=True) if isinstance(pred, torch.Tensor) else pred for pred in predictions]

        rouge_scores = [ROUGE_L(pred, ref) for pred, ref in zip(predictions, targets)]
        rdass_scores = [RDASS(doc, ref, pred, embedding_model, tokenizer) for doc, ref, pred in zip(documents, targets, predictions)]

        for doc, pred, ref in zip(documents, predictions, targets):
            print("-- Inputs --")
            print(doc)
            print("-- Model Predictions --")
            print(pred)
            print("-- Answer --")
            print(ref)
        
        print("ROUGE scores:", rouge_scores)
        print("RDASS scores:", rdass_scores)

gc.collect()
torch.cuda.empty_cache()
