import os
import pandas as pd
from typing import Optional
from tqdm.auto import tqdm
from datasets import Dataset, DatasetDict
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def from_processed(dir, predict=False):
    df = pd.read_csv(dir)
    text_lst = []
    lable_lst = []
    for _, item in tqdm(df.iterrows(), desc='tokenizing', total=len(df)):
        text = '[SEP]'.join([item[text_column].strip() for text_column in ['sentence_1', 'sentence_2']])
        text_lst.append(text)
        if predict:
            lable_lst.append(0)
        else:
            lable_lst.append(item['label'])
            
    dataset = Dataset.from_dict({"text": text_lst, "labels": lable_lst})
    return dataset


class STSDataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, max_length, num_workers, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.num_workers = num_workers
        
        self.dsdict = DatasetDict()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_length)

    def setup(self, stage=None):
        if stage == "fit":
            self.dsdict["train"] = from_processed(self.train_path)
            self.dsdict["validation"] = from_processed(self.dev_path)

        elif stage == "test":
            self.dsdict["test"] = from_processed(self.dev_path)

        else:
            self.dsdict["test"] = from_processed(self.test_path, predict=True)
            
    def _preprocess(self, batch: dict) -> dict:
        tokens = self.tokenizer(
            batch["text"], 
            add_special_tokens=True, 
            padding='max_length', 
            truncation=True
        )
        tokens["label"] = [[label] for label in batch["labels"]]
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
        tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
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
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._shared_transform("test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self._shared_transform("test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


def from_processed_SimCSE(dir, predict=False):
    df = pd.read_csv(dir)
    sentence_lst = []
    for _, row in df.iterrows():
        sentence_lst.append(row['sentence_1'])
        sentence_lst.append(row['sentence_2'])
    dataset = Dataset.from_dict({"text": sentence_lst})
    
    return dataset


class SimCSEDataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, max_length, num_workers, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path
        
        self.dsdict = DatasetDict()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_length)

    def setup(self, stage=None):
        if stage == "fit":
            self.dsdict["train"] = from_processed_SimCSE(self.train_path)
            self.dsdict["validation"] = from_processed_SimCSE(self.dev_path)

        elif stage == "test":
            self.dsdict["test"] = from_processed_SimCSE(self.dev_path)

        else:
            self.dsdict["test"] = from_processed_SimCSE(self.test_path)
            
    def _preprocess(self, batch: dict) -> dict:
        tokens = self.tokenizer(
            batch["text"], 
            add_special_tokens=True, 
            padding='max_length', 
            truncation=True
        )
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
            type="torch", columns=["input_ids", "attention_mask"]
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
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._shared_transform("test"),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
