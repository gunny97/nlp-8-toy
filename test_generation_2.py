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
# from huggingface_hub import login

# login()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def predict_loop(config):
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
 # 
 # 'epoch=12-Val_Loss=1.53.ckpt'
    trainer = Trainer(
        callbacks=callbacks,
        devices=1,
        accelerator="gpu",
        max_epochs=config.epochs,
        accumulate_grad_batches=4,
        
    )

    predictions = trainer.predict(model=model, datamodule=dm, ckpt_path = 'resources/log/gen/epoch=3-Val_Loss=1.56.ckpt')
    print(predictions)
    exit()



if __name__ == "__main__":

    torch.cuda.empty_cache()
    gc.collect()

    train_config = add_options()

    # Train the model
    predict_loop(train_config)
