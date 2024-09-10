from typing import Any
import numpy as np
from utils.scheduler import WarmupDecayLR
import torch
from utils.helpers import find_linear_names
from pytorch_lightning import LightningModule
from peft import get_peft_model, LoraConfig
from torch.optim import AdamW, Optimizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class TransformerModule(LightningModule):

    def __init__(
        self,
        pretrained_model: str,
        lr: float,
    ):
        super().__init__()
        self.lr = lr

        # Configuration for LoRA and quantization using BitsAndBytes
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model,
            device_map="cuda:0",
            quantization_config=bnb_config
        )
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        # LoRA configuration
        peft_config = LoraConfig(
            r=6,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "o_proj", "k_proj", "v_proj", 
                "gate_proj", "up_proj", "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        """
        Training step for calculating loss and logging metrics.
        """
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss = outputs.loss
        
        self.log("Train_Loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for calculating and logging validation loss.
        """
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        val_loss = outputs.loss

        self.log("Val_Loss", val_loss, logger=True)
        return val_loss

    def configure_optimizers(self) -> Optimizer:
        """
        Set up the optimizer (AdamW) for training with learning rate scheduling.
        """
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.98))
        scheduler = get_linear_schedule_with_warmup(
		        optimizer,
		        num_warmup_steps=1000,  # You can adjust the warmup steps based on your dataset
		        num_training_steps=self.trainer.estimated_stepping_batches,  # Total training steps
		    )

        return [optimizer], [scheduler]