import gc
import os
import torch
import random
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from model.fine_tune_sts import TransformerModule
from utils.data_module import STSDataModule
from pytorch_lightning.loggers import WandbLogger

# seed 고정
torch.manual_seed(104)
torch.cuda.manual_seed(104)
torch.cuda.manual_seed_all(104)
random.seed(104)


def training_loop(config):
    model = TransformerModule(
        model_name=config.model_name,
        lr=config.lr,
        dataset_size=config.dataset_size,
        max_epoch=config.max_epoch,
        batch_size=config.batch_size,
        warmup=config.warmup,
        beta1=config.beta1,
        beta2=config.beta2,
        weight_decay=config.weight_decay,
        CL=config.CL,
    )

    dm = STSDataModule(
        model_name=config.model_name,
        batch_size=config.batch_size,
        max_length=config.max_length,
        num_workers=config.num_workers,
        train_path=config.train_path,
        dev_path=config.dev_path,
        test_path=config.test_path,
        predict_path=config.predict_path
    )

    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{config.pretrained_model}_{config.target_name}_{config.exp_name}",
            project="STS",
            save_dir=config.save_dir + 'wandb',
            log_model=True,
        )
        wandb_logger.log_hyperparams(config)

    checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir + 'ckpt',
            filename=f"{args.warmup}_{args.beta1}_{args.beta2}_{args.weight_decay}_"+"{epoch}-{val_pearson:.3f}",
            monitor="val_pearson",
            mode="max",
            verbose=True,
            save_top_k=1,
        )
    
    # Run the training loop.
    trainer = Trainer(accelerator="gpu", devices=1, 
                    max_epochs=args.max_epoch, callbacks=checkpoint_callback, 
                    gradient_clip_val=0.0, log_every_n_steps=1,
                    logger=wandb_logger if args.wandb else None
                    )
        

    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)
    
if __name__ == "__main__":
    # Free up gpu vRAM from memory leaks.
    torch.cuda.empty_cache()
    gc.collect()

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deliciouscat/kf-deberta-base-cross-sts', type=str, choices=['deliciouscat/kf-deberta-base-cross-sts'])
    parser.add_argument('--train_path', default='./resources/raw/train.csv')
    parser.add_argument('--dev_path', default='./resources/raw/dev.csv')
    parser.add_argument('--test_path', default='./resources/raw/dev.csv')
    parser.add_argument('--predict_path', default='./resources/raw/test.csv')
    parser.add_argument('--CL', default=None, help='Load Unsupervised Contrastive Learning Model')
    parser.add_argument('--save_dir', default='./resources/log/raw/')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--dataset_size', default=9324, type=int, choices=[9324])
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument("--warmup", type=int, default=350, help="Number of warmup steps")
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument('--wandb', default=False, type=float)

    args = parser.parse_args()
    
    # Train model.
    training_loop(args)
