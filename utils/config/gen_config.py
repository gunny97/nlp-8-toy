import os
import argparse
from argparse import ArgumentParser


def add_options():
    # fmt: off
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--train_data", default="resources/gen/gen_train.csv", type=str)
    parser.add_argument("--test_data", default="resources/gen/gen_valid.csv", type=str)

    # Model
    parser.add_argument("--pretrained_model", default="google/gemma-2b-it", type=str, choices=['beomi/gemma-ko-2b', 'google/gemma-2b-it'])

    # Training
    parser.add_argument("--num_workers", default=os.cpu_count(), type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    # parser.add_argument("--min_delta", default=0.05, type=int)
    # parser.add_argument("--patience", default=6, type=int)
    parser.add_argument("--model_checkpoint_dir", default="resources/log/gen", type=str)
    parser.add_argument("--exp_name", default="generation", type=str)

    args = parser.parse_args()
    # fmt: on
    return args
