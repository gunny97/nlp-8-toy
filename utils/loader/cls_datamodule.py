import os
import pandas as pd
from typing import Optional
from datasets import Dataset, DatasetDict
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def from_processed(dir: str, target_name: str):
    df = pd.read_csv(dir)
    df[target_name] = df[target_name].astype("category")
    df[target_name] = df[target_name].cat.codes
    dataset = Dataset.from_dict(
        {"text": df["utterances_text"].tolist(), "labels": df[target_name].tolist()}
    )
    # print("데이터 확인 : ", dataset[0])
    
    # dataset_dict = {
    #     "text" : df["utterances_text"].tolist(),
    # }
    # if "topic" in df.columns:
    #     dataset_dict["topic"] = df["topic"].astype("category").cat.codes.tolist()
    # if "keyword" in df.columns:
    #     dataset_dict["keyword"] = df["keyword"].astype("category").cat.codes.tolist()
    # if "speech_act" in df.columns:
    #     dataset_dict["speech_act"] = df["speech_act"].astype("category").cat.codes.tolist()
        
    # dataset = Dataset.from_dict(dataset_dict)
    
    return dataset


class ClsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: str,
        val_data: str,
        train_act_data: str,
        val_act_data: str,
        #test_data: str,
        target_name: str,
        pretrained_model: str,
        max_length: int,
        batch_size: int,
        num_workers: Optional[int] = 4,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_act_data = train_act_data
        self.val_act_data = val_act_data
        self.target_name = target_name
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dsdict = DatasetDict()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.target_name in ["topic", "keyword"]:
                self.dsdict["train"] = from_processed(self.train_data, self.target_name)
                self.dsdict["validation"] = from_processed(self.val_data, self.target_name)
            elif self.target_name == "speech_act":
                self.dsdict["train"] = from_processed(self.train_act_data, self.target_name)
                self.dsdict["validation"] = from_processed(self.val_act_data, self.target_name)


        if stage == "test" or stage is None:
            if self.target_name in ["topic", "keyword"]:
                self.dsdict["test"] = from_processed(self.val_data, self.target_name)
            elif self.target_name == "speech_act":
                self.dsdict["test"] = from_processed(self.val_act_data, self.target_name)

    def _preprocess(self, batch: dict) -> dict:
        tokens = self.tokenizer(
            batch["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        #tokens["labels"] = torch.tensor(batch["labels"], dtype=torch.long)
        tokens["labels"] = batch["labels"]
        return tokens

    def _shared_transform(self, split: str) -> torch.tensor:
        """
        Tokenize the given split, and then convert from arrow to pytorch tensor format.
        """
        ds = self.dsdict[split]
        tokenized_ds = ds.map(
            self._preprocess,
            batched=True,
            load_from_cache_file=True,
        )
        # print("데이터 확인 2: ", tokenized_ds[0])
        tokenized_ds.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return tokenized_ds

    def train_dataloader(self):
        return DataLoader(
            dataset=self._shared_transform("train"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._shared_transform("validation"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._shared_transform("test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )
