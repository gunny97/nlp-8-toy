import torch
from pytorch_lightning import LightningModule
import torchmetrics
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification


class TransformerModule(LightningModule):
    def __init__(self, model_name, lr, dataset_size, max_epoch, batch_size, warmup, beta1, beta2, weight_decay, CL):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.dataset_size = dataset_size
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.warmup = warmup
        self.beta1 = beta1 
        self.beta2 = beta2
        self.weight_decay = weight_decay
        
        if CL:
            # contrastive learning 한 체크포인트 불러오기
            ckpt = torch.load(CL)
            
            # 체크포인트에서 모델 가중치 추출
            state_dict = ckpt["state_dict"]
            
            # 백본 가중치만 추출
            backbone_state_dict = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}

            self.plm = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, 
                                                                            state_dict=backbone_state_dict, num_labels=1)
        else:
            self.plm = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,num_labels=1)
            
        self.loss_func = torch.nn.SmoothL1Loss()

    def forward(self, input_ids, attention_mask):
        x = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return x['logits']

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch["input_ids"], batch["attention_mask"], batch["label"]
        logits = self(input_ids, attention_mask)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch["input_ids"], batch["attention_mask"], batch["label"]
        logits = self(input_ids, attention_mask)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch["input_ids"], batch["attention_mask"], batch["label"]
        logits = self(input_ids, attention_mask)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        logits = self(input_ids, attention_mask)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        num_steps = int(self.dataset_size * self.max_epoch / self.batch_size)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup, num_training_steps=num_steps)
        lr_scheduler = {'scheduler': scheduler, 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler] 
