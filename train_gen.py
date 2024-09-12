import gc
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model.fine_tune_gen import TransformerModuleForGen
from utils.loader.gen_datamodule import GenDataModule
from pytorch_lightning.loggers import WandbLogger
from utils.config.gen_config import add_options
import os
from huggingface_hub import login

login()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def training_loop(config):
    """
    Training loop that sets up data module, logger, and trainer for training the model.
    """
    model = TransformerModuleForGen(
        pretrained_model=config.pretrained_model,
        lr=config.lr,
        max_length=config.max_length,
    )

    dm = GenDataModule(
        train_data=config.train_data,
        valid_data=config.test_data,
        pretrained_model=config.pretrained_model,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    wandb_logger = WandbLogger(
        name=f"{config.pretrained_model}_{config.exp_name}",
        project="nlp_gen",
        save_dir=config.model_checkpoint_dir,
        log_model=True,
    )
    wandb_logger.log_hyperparams(config)

    # Checkpointing and early stopping based on validation loss
    callbacks = [
        ModelCheckpoint(
            dirpath=config.model_checkpoint_dir,
            filename="{epoch}-{Val_Loss:.2f}",
            monitor="Val_Loss",
            mode="min",
            save_top_k=1,
        )
    ]

    trainer = Trainer(
        callbacks=callbacks,
        devices=1,
        accelerator="gpu",
        max_epochs=config.epochs,
        accumulate_grad_batches=4,
        logger=wandb_logger,
    )

    # Training
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":

    torch.cuda.empty_cache()
    gc.collect()

    train_config = add_options()

    # Train the model
    training_loop(train_config)
