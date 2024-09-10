import gc
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model.fine_tune_gen import TransformerModule
# from utils.loader.gen_datamodule import CausalDataModule
from pytorch_lightning.loggers import WandbLogger
# from utils.config.gen_config import add_options


def training_loop(config):
    """
    Training loop that sets up data module, logger, and trainer for training the model.
    """
    model = TransformerModule(
        pretrtrained_model=config.pretrained_model,
        lr=config.lr,
    )

    dm = CausalDataModule(
        train_data=config.train_data,
        valid_data=config.valid_data,
        tokenizer=model.tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    wandb_logger = WandbLogger(
        name=f"{config.pretrained_model}_{config.exp_name}",
        project="nlp_generation",
        save_dir=config.model_checkpoint_dir,
        log_model=True,
    )

    # Checkpointing and early stopping based on validation loss
    callbacks = [
        ModelCheckpoint(
            dirpath=config.model_checkpoint_dir,
            filename="{epoch}-{Val_Loss:.2f}",
            monitor="Val_Loss",
            mode="min",
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="Val_Loss",
            min_delta=config.min_delta,
            patience=config.patience,
            mode="min",
        ),
    ]

    trainer = Trainer(
		callbacks=callbacks,
		devices=1,
		accelerator="gpu",
        max_epochs=config.epochs,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=wandb_logger,
    )

    # Training
    trainer.fit(model=model, datamodule=dm)

    # Save the fine-tuned model and LoRA adapter
    # model.model.save_pretrained("lora_adapter_dir", save_adapter=True)

if __name__ == "__main__":

    torch.cuda.empty_cache()
    gc.collect()

    train_config = add_options()

    # Train the model
    trained_model, data_module = training_loop(train_config)