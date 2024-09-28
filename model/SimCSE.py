import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup


class Model(LightningModule):
    def __init__(self, model_name, lr, temperature, dataset_size, max_epoch, batch_size, warmup, beta1, beta2, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.temperature = temperature
        self.dataset_size = dataset_size
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.warmup = warmup
        self.beta1 = beta1 
        self.beta2 = beta2
        self.weight_decay = weight_decay
    
        # you can use any models
        self.backbone = AutoModel.from_pretrained(self.model_name)
        
        # define additional MLP layer
        # see Section 6.3 of the paper for more details
        # refenrece: https://github.com/princeton-nlp/SimCSE/blob/511c99d4679439c582beb86a0372c04865610b6b/simcse/models.py#L19
        self.hidden_size = self.backbone.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return outputs

    def _compute_loss(self, emb1, emb2) -> tuple:
        # SimCSE training objective:
        #    maximize the similarity between the same sentence
        # => make diagonal elements most similar

        # shape of sim_matrix: (batch_size, batch_size)
        # calculate cosine similarity between all pair of embeddings (n x n)
        sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
        # FYI: SimCSE is sensitive for the temperature parameter.
        # see Table D.1 of the paper
        sim_matrix = sim_matrix / self.temperature
        
        # labels := [0, 1, 2, ..., batch_size - 1]
        # labels indicate the index of the diagonal element (i.e. positive examples)
        labels = torch.arange(emb1.shape[0]).long().to("cuda")
        
        # it may seem strange to use Cross-Entropy Loss here.
        # this is a shorthund of doing SoftMax and maximizing the similarity of diagonal elements
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
        
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

        # original SimCSE uses MLP layer only during training
        # see: Table 6 of the paper
        # this trick is a bit complicated, so you may omit it when training your own model
        # different dropout masks are adapt automatically
        outputs1 = self(input_ids, attention_mask)
        
        # take representations of [CLS] token
        # we only implement the best performing pooling, [CLS], for simplicity
        # you can easily extend to other poolings (such as mean pooling or max pooling) by edting this line
        # shape of last_hidden_state: (batch_size, seq_len, hidden_size)
        emb1 = outputs1.last_hidden_state[:, 0]
        emb1 = self.dense(emb1)
        emb1 = self.activation(emb1)

        # simply forward inputs twice!
        # different dropout masks are adapt automatically
        outputs2 = self(input_ids, attention_mask)
        emb2 = outputs2.last_hidden_state[:, 0]
        emb2 = self.dense(emb2)
        emb2 = self.activation(emb2)

        loss = self._compute_loss(emb1, emb2)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        
        outputs1 = self(input_ids, attention_mask)
        emb1 = outputs1.last_hidden_state[:, 0]
        emb1 = self.dense(emb1)
        emb1 = self.activation(emb1)

        outputs2 = self(input_ids, attention_mask)
        emb2 = outputs2.last_hidden_state[:, 0]
        emb2 = self.dense(emb2)
        emb2 = self.activation(emb2)

        loss = self._compute_loss(emb1, emb2)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        
        outputs1 = self(input_ids, attention_mask)
        emb1 = outputs1.last_hidden_state[:, 0]
        emb1 = self.dense(emb1)
        emb1 = self.activation(emb1)

        outputs2 = self(input_ids, attention_mask)
        emb2 = outputs2.last_hidden_state[:, 0]
        emb2 = self.dense(emb2)
        emb2 = self.activation(emb2)

        loss = self._compute_loss(emb1, emb2)
        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        num_steps = int(self.dataset_size * self.max_epoch / self.batch_size)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup, num_training_steps=num_steps)
        lr_scheduler = {'scheduler': scheduler, 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler] 
