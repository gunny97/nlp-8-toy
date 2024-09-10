import os
import pandas as pd
from typing import Optional
from datasets import Dataset, DatasetDict
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def from_processed(dir: str):
    df = pd.read_csv(dir)
    prompt_list = []
    for i in range(len(df['text'])):
        prompt_list.append(
            f"<bos><start_of_turn>user\n다음 글을 요약해주세요:\n"
            f"{df['text'][i]}<end_of_turn>\n<start_of_turn>model\n"
            f"{df['summarized_text'][i]}<end_of_turn><eos>"
        )
    dataset = Dataset.from_dict(
        {"text": prompt_list, "summarized_text": df["utterances_summarize_result"].tolist()}
    )
    return dataset


class GenDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: str,
        valid_data: str,
        pretrained_model: str,
        max_length: int,
        batch_size: int,
        num_workers: Optional[int] = 4,
    ):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dsdict = DatasetDict()
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dsdict["train"] = from_processed(self.train_data)
            

        if stage == "test" or stage is None:
            self.dsdict["validation"] = from_processed(
                self.valid_data
            )

    def _preprocess(self, batch: dict) -> dict:
        tokens = self.tokenizer(
            batch["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        tokens["label"] = [label for label in batch["utterances_summarize_result"]]
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
            type="torch", columns=["input_ids", "attention_mask", "label"]
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

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self._shared_transform("test"),
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         drop_last=True,
    #     )
