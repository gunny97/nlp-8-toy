from typing import Any
import numpy as np
from utils.scheduler import WarmupDecayLR
import torch
from utils.helpers import find_linear_names
from pytorch_lightning import LightningModule
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import AdamW, Optimizer
from transformers import get_linear_schedule_with_warmup
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


class TransformerModule(LightningModule):
    """
    Takes a pretrained model with classification head and uses the peft package to do Adapter + LoRA fine tuning.
    """

    def __init__(
        self,
        pretrained_model: str,
        num_classes: int,
        lr: float,
    ):
        super().__init__()
        # # 양자화를 사용해서 학습 시 loss가 Nan이 나오는 경우가 있어요.
        # # 그 이유는 저도 자세히는 잘 모르겠지만 성능 측면에서는 양자화를 사용하지 않는게 좋으니 빼서 사용했어요!
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        self.num_classes = num_classes
        self.lr = lr
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            num_labels=self.num_classes,
            # quantization_config=bnb_config,
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,  # Dropout rate for the adapter
            bias="none",  # Bias configuration for the adapter
            r=32,  # 8
            lora_alpha=8,
        )
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()

    def forward(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        label: list[int],
    ):
        """Calc the loss by passing inputs to the model and comparing against ground
        truth labels. Here, all of the arguments of self.model comes from the
        SequenceClassification head from HuggingFace.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label,
        )

    def _compute_metrics(self, pred_class, label, prefix) -> tuple:
        metrics = {
            f"{prefix}_Acc": multiclass_accuracy(
                preds=pred_class, target=label, num_classes=self.num_classes
            ),
            f"{prefix}_F1_Score": multiclass_f1_score(
                preds=pred_class,
                target=label,
                num_classes=self.num_classes,
                average="macro",
            ),
            f"{prefix}_Precision": multiclass_precision(
                preds=pred_class,
                target=label,
                num_classes=self.num_classes,
                average="macro",
            ),
            f"{prefix}_Recall": multiclass_recall(
                preds=pred_class,
                target=label,
                num_classes=self.num_classes,
                average="macro",
            ),
        }

        return metrics

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            label=batch["label"],
        )
        self.lr_schedulers().step()

        # For predicting probabilities, do softmax along last dimension (by row).
        pred_class = torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1)
        # Calculate Score
        metrics = self._compute_metrics(pred_class, batch["label"], "Train")

        # wandb logging
        self.log("Train_Loss", outputs["loss"], logger=True)
        for k, v in metrics.items():
            self.log(f"{k}", v, logger=True, on_epoch=True, on_step=False)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx) -> dict[str, Any]:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            label=batch["label"],
        )
        # For predicting probabilities, do softmax along last dimension (by row).
        pred_class = torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1)
        # Calculate Score
        metrics = self._compute_metrics(pred_class, batch["label"], "Val")

        # wandb logging
        self.log("Val_Loss", outputs["loss"], logger=True, on_epoch=True, on_step=False)
        for k, v in metrics.items():
            self.log(f"{k}", v, logger=True, on_epoch=True, on_step=False)
        return outputs["loss"]

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            label=batch["label"],
        )
        # For predicting probabilities, do softmax along last dimension (by row).
        pred_class = torch.argmax(torch.softmax(outputs["logits"], dim=-1), dim=1)
        # Calculate Score
        metrics = self._compute_metrics(pred_class, batch["label"], "Test")

        # wandb logging
        self.log("Test_Loss", outputs["loss"])
        for k, v in metrics.items():
            self.log(f"{k}", v)
        return outputs["loss"]

    # def configure_optimizers(self) -> Optimizer:
    #     optimizer = AdamW(
    #         self.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.98)
    #     )
    #     scheduler = WarmupDecayLR(optimizer, warmup_steps=10000, d_model=512)
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        optimizer = AdamW(
            self.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.98)
        )
        num_steps = int(9974  * 10 / 16)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=300, num_training_steps=num_steps)
        lr_scheduler = {'scheduler': scheduler, 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]