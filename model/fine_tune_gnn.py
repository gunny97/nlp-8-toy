import torch
import torchmetrics
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel
from pytorch_lightning import LightningModule
from utils.helpers import load_pickle


class TransformerWithGNN(LightningModule):
    def __init__(self, model_name, lr, dataset_size, max_epoch, batch_size, warmup, beta1, beta2, weight_decay):
        '''시간이 없어 최종적으로 사용하지 못해 다양한 실험을 할 수 없었음. 빠르게 구현하기 위해 하드코딩된 부분이 일부 있음
        '''
        
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
        
        
        # 노드의 워드벡터 불러오기
        self.node_init_vec = torch.tensor(load_pickle("word2vec.pkl")[1:, :], dtype=torch.float).to("cuda")

        # 인접 행렬을 sparse 형태로 변환하여 edge_index 생성
        adjacency_matrix = torch.tensor(load_pickle("adj_matrix.pkl")[1:, 1:], dtype=torch.float)
        self.edge_index, _ = dense_to_sparse(adjacency_matrix)
        self.edge_index = self.edge_index.to("cuda")
        
        # 인접 행렬 가중치 가져오기
        edge_weight = adjacency_matrix[adjacency_matrix != 0]
        self.edge_weight = edge_weight.view(-1, 1).to("cuda")

        # Self-adaptive Adjacency Matrix 생성
        self.nodevec1 = nn.Parameter(torch.randn(1000, 10).to("cuda"), requires_grad=True).to("cuda")
        self.nodevec2 = nn.Parameter(torch.randn(10, 1000).to("cuda"), requires_grad=True).to("cuda")
        
        # 문맥을 임베딩
        self.plm = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        
        # 단어 간의 관계를 임베딩
        self.gcn1 = GCNConv(768, 1024)  # 첫 번째 GCNConv 레이어
        self.gcn2 = GCNConv(1024, 768)  # 첫 번째 GCNConv 레이어
        
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.regressor = nn.Linear(in_features=1000, out_features=1, bias=True)
        self._initialize_weights(self.regressor)
        
    def _initialize_weights(self, layer):
        # Xavier Uniform 초기화
        init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)
            
    def forward(self, input_ids, attention_mask):
        # Self-adaptive Adjacency Matrix 계산
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adp = adp.view(-1, 1)
        edge_adp_weight = self.edge_weight + adp

        # 단어 간 관게 임베딩
        graph_x = self.gcn1(x = self.node_init_vec, edge_index = self.edge_index, edge_weight=self.edge_weight + edge_adp_weight)
        graph_x = F.relu(graph_x)
        graph_x = self.gcn2(x = graph_x, edge_index = self.edge_index, edge_weight=self.edge_weight) # (노드 개수, 768)
        graph_x = graph_x.transpose(0, 1) # (768, 노드 개수)

        # 토큰 별 히든 벡터와 그래프 임베딩 내적
        x = self.plm(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'] # (32, 512(토큰 수), 768)
        x_matmul = torch.matmul(x, graph_x) # (32, 512(토큰 수), 노드 개수)
        
        # 평균 풀링 (문장 단위의 벡터로 변환)
        x_pooled = torch.mean(x_matmul, dim=1)  # (32, 노드 개수)
        
        # Dropout 레이어 적용
        x_pooled = self.dropout(x_pooled)
        
        # Linear 레이어 적용
        logits = self.regressor(x_pooled)

        return logits


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
