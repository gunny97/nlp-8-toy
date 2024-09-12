import gc
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from model.fine_tune_cls import TransformerModule
from utils.loader.cls_datamodule import ClsDataModule
from pytorch_lightning.loggers import WandbLogger
from utils.config.cls_config import add_options


def training_loop(config) -> TransformerModule:
    
    if config.target_name == 'topic':
        num_classes = 9
        train_data = config.train_data
        val_data = config.val_data
        
    elif config.target_name == 'keyword':
        num_classes = 87
        train_data = config.train_data
        val_data = config.val_data
        
    elif config.target_name == 'speech_act':
        train_data = config.train_act_data
        val_data = config.val_act_data
        num_classes = 3
    else:
        raise ValueError(f"invalid target_name : {config.target_name}")
    
    
    model = TransformerModule(
        pretrained_model=config.pretrained_model,
        num_classes=num_classes,
        lr=config.lr,
        use_quantization=config.use_quantization,
    )
    dm = ClsDataModule(
        train_data=train_data,
        val_data=val_data,
        train_act_data=config.train_act_data,
        val_act_data=config.val_act_data,
        #test_data=config.test_data,
        target_name=config.target_name,
        pretrained_model=config.pretrained_model,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    wandb_logger = WandbLogger(
        name=f"{config.pretrained_model}_{config.target_name}_{config.exp_name}",
        project="nlp",
        save_dir=config.model_checkpoint_dir,
        log_model=True,
    )
    wandb_logger.log_hyperparams(config)

    # Keep the model with the highest F1 score.
    callbacks = [
        ModelCheckpoint(
            dirpath=config.model_checkpoint_dir,
            filename="{epoch}-{Val_F1_Score:.2f}",
            monitor="Val_F1_Score",
            mode="max",
            verbose=True,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="Val_F1_Score",
            min_delta=config.min_delta,
            patience=config.patience,
            mode="max",
        ),
    ]

    # Run the training loop.
    trainer = Trainer(
        callbacks=callbacks,
        default_root_dir=config.model_checkpoint_dir,
        devices=1,  # 특정 gpu에 할당, 새로운 서버에는 0으로 초기화 필요
        accelerator="gpu",
        max_epochs=config.epochs,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=wandb_logger,
    )
    trainer.fit(model=model, datamodule=dm)

    # # Evaluate the last and the best models on the test sample.
    # best_model_path = checkpoint_callback.best_model_path
    # trainer.test(model=model, datamodule=datamodule)
    # trainer.test(
    #     model=model,
    #     datamodule=datamodule,
    #     ckpt_path=best_model_path,
    # )


if __name__ == "__main__":

    # Free up gpu vRAM from memory leaks.
    torch.cuda.empty_cache()
    gc.collect()

    train_config = add_options()

    # Train model.
    training_loop(train_config)
