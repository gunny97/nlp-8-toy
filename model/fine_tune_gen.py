import wandb
from typing import Any
import numpy as np
from utils.scheduler import WarmupDecayLR
import torch
from utils.helpers import find_linear_names
from pytorch_lightning import LightningModule
from peft import get_peft_model, LoraConfig

# from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
from torch.optim import Optimizer, AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from bitsandbytes.optim import PagedAdamW
import nltk

nltk.download("punkt_tab")


class TransformerModuleForGen(LightningModule):
    def __init__(self, pretrained_model: str, lr: float, max_length: int):
        super().__init__()
        self.lr = lr
        self.max_length = max_length

        # Configuration for LoRA and quantization using BitsAndBytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            quantization_config=bnb_config,
        )

        # modules  = find_linear_names(model, 'qlora')

        # LoRA configuration
        peft_config = LoraConfig(
            r=32,
            lora_alpha=8,
            lora_dropout=0.05,
            # target_modules=modules,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()

        # Get Evlaudation Metric
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        # self.bertscore = BERTScore() # 편의를 위해 사용, 한국어 BERT 모델로 변경 필요
        self.rouge = ROUGEScore()

    def forward(self, input_ids, attention_mask):
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        """
        Training step for calculating loss and logging metrics.
        """
        outputs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        self.log("Train_Loss", outputs["loss"], logger=True)
        return outputs["loss"]

    def _compute_metrics(self, pred, target, prefix) -> tuple:
        rouge_lst = [
            self.rouge(p, t)["rouge1_fmeasure"].item() for p, t in zip(pred, target)
        ]
        self.log(f"{prefix}_ROUGE", np.mean(rouge_lst), logger=True)

        # Logging Examples
        columns = ["{prefix}_Target_Sample", "{prefix}_Pred_Sample"]
        data = list(zip(target[:3], pred[:3]))
        table = wandb.Table(data=data, columns=columns)
        wandb.log({f"examples/{prefix}": table})

    def validation_step(self, batch, batch_idx):
        """
        Validation step for calculating and logging validation loss.
        """
        outputs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # print('input: ', batch['input'][0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs_ = self.tokenizer(
            batch["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        preds = self.model.generate(**inputs_, max_new_tokens=64, use_cache=True)
        preds = [
            self.tokenizer.decode(pred)
            .split("<start_of_turn>model")[-1]
            .split("<end_of_turn>")[0]
            .strip()
            for pred in preds
        ]
        # print('pred: ', preds[0])
        # exit()

        # wandb logging
        self.log("Val_Loss", outputs["loss"], logger=True)
        self._compute_metrics(preds, batch["label"], "Val")

        return outputs["loss"]

    def test_step(self, batch, batch_idx):
        """
        Validation step for calculating and logging validation loss.
        """
        outputs = self(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # print('input: ', batch['input'][0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs_ = self.tokenizer(
            batch["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        preds = self.model.generate(**inputs_, max_new_tokens=64, use_cache=True)
        preds = [
            self.tokenizer.decode(pred)
            .split("<start_of_turn>model")[-1]
            .split("<end_of_turn>")[0]
            .strip()
            for pred in preds
        ]
        # print('pred: ', preds[0])

        # wandb logging
        self.log("Test_Loss", outputs["loss"], logger=True)
        self._compute_metrics(preds, batch["label"], "Test")

        return outputs["loss"]

    def configure_optimizers(self) -> Optimizer:
        """
        Set up the optimizer (AdamW) for training with learning rate scheduling.
        """
        optimizer = PagedAdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        # optimizer = AdamW(
        #     self.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.98)
        # )
        scheduler = WarmupDecayLR(optimizer, warmup_steps=10, d_model=512)

        return [optimizer], [scheduler]
