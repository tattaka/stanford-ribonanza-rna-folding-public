import argparse
import datetime
import gc
import math
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, Dataset
from train import (
    EXP_ID,
    SEQ_PATH,
    RibonanzaDataModule,
    RibonanzaDataset,
    RibonanzaLightningModel,
    get_optimizerd_dtype,
)
from transformers import get_cosine_schedule_with_warmup

EXP_ID = f"{EXP_ID}_finetune"


class RibonanzaDataFTModule(RibonanzaDataModule):
    def create_dataset(self, mode: str = "train") -> RibonanzaDataset:
        if mode == "train":
            train_dataset = RibonanzaDataset(
                df=self.df,
                split=self.train_split,
                mode=mode,
                mask_only=False,
                filter_rate=1,
            )
            train_dataset_mask = RibonanzaDataset(
                df=self.df,
                split=self.train_split,
                mode=mode,
                mask_only=True,
                filter_rate=1,
            )
            pl_split = np.random.permutation(len(self.pseudo_label_df) // 2)[
                : len(train_dataset)
            ]
            pl_dataset = RibonanzaDataset(
                df=self.pseudo_label_df,
                split=pl_split,
                mode=mode,
                mask_only=False,
                filter_rate=1,
            )
            pl_dataset_mask = RibonanzaDataset(
                df=self.pseudo_label_df,
                split=pl_split,
                mode=mode,
                mask_only=True,
                filter_rate=1,
            )
            return (
                ConcatDataset(
                    [
                        train_dataset,
                        pl_dataset,
                    ]
                ),
                ConcatDataset(
                    [
                        train_dataset_mask,
                        pl_dataset_mask,
                    ]
                ),
            )
        else:
            return (
                RibonanzaDataset(
                    df=self.df,
                    split=self.val_split,
                    mode=mode,
                    mask_only=False,
                ),
                RibonanzaDataset(
                    df=self.df,
                    split=self.val_split,
                    mode=mode,
                    mask_only=True,
                ),
            )


class RibonanzaLightningFTModel(RibonanzaLightningModel):
    def configure_optimizers(self):
        # self.warmup = True
        self.warmup = False
        optimizer = AdamW(
            self.get_optimizer_parameters(),
            eps=1e-6,
        )
        max_train_steps = self.trainer.estimated_stepping_batches
        warmup_steps = math.ceil((max_train_steps * 2) / 100) if self.warmup else 0
        rank_zero_info(
            f"max_train_steps: {max_train_steps}, warmup_steps: {warmup_steps}"
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]


def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--seed",
        default=2022,
        type=int,
        metavar="SE",
        help="seed number",
        dest="seed",
    )
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="1 batch run for debug",
        dest="debug",
    )
    parent_parser.add_argument(
        "--no_amp",
        action="store_true",
        help="not using amp",
        dest="no_amp",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--resumedir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parent_parser.add_argument(
        "--pseudo_label_df",
        default=None,
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parent_parser.add_argument(
        "--gpus", type=int, default=4, help="number of gpus to use"
    )
    parent_parser.add_argument(
        "--epochs", default=30, type=int, metavar="N", help="total number of epochs"
    )
    parser = RibonanzaLightningFTModel.add_model_specific_args(parent_parser)
    parser = RibonanzaDataFTModule.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    pl.seed_everything(args.seed, workers=True)
    if not args.debug:
        warnings.simplefilter("ignore")
    pseudo_label_df = pd.read_csv(args.pseudo_label_df, nrows=5)
    key_to_dtype = get_optimizerd_dtype(pseudo_label_df)
    key_to_dtype = {
        key: key_to_dtype[key] for key in key_to_dtype if "error" not in key
    }
    pseudo_label_df = pd.read_csv(
        args.pseudo_label_df, usecols=key_to_dtype.keys(), dtype=key_to_dtype
    )
    df = pd.read_parquet(
        os.path.join(SEQ_PATH, "train_data_kmeans_groupkfold_structures.parquet")
    )
    key_to_dtype = get_optimizerd_dtype(df)
    key_to_dtype = {
        key: key_to_dtype[key] for key in key_to_dtype if "error" not in key
    }
    df = df.loc[:, key_to_dtype.keys()].astype(key_to_dtype)
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    assert args.fold < 5
    for fold, (train_split, val_split) in enumerate(kf.split(np.arange(len(df) // 2))):
        if args.fold != fold:
            continue
        datamodule = RibonanzaDataFTModule(
            df=df,
            pseudo_label_df=pseudo_label_df,
            train_split=train_split,
            val_split=val_split,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        resume_checkpoint = glob(
            f"../../logs/{args.resumedir}/fold{fold}/**/best_loss.ckpt", recursive=True
        )[0]
        print(f"load checkpoint: {resume_checkpoint}")
        model = RibonanzaLightningFTModel.load_from_checkpoint(
            resume_checkpoint, lr=args.lr
        )
        logdir = f"../../logs/exp{EXP_ID}/{args.logdir}/fold{fold}"

        print(f"logdir = {logdir}")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename="best_loss",
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", log_rank_zero_only=True
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        if not args.debug:
            wandb_logger = WandbLogger(
                name=f"exp{EXP_ID}/{args.logdir}/fold{fold}",
                save_dir=logdir,
                project="stanford-ribonanza-rna-folding",
                tags=["finetune"],
            )

        trainer = pl.Trainer(
            default_root_dir=logdir,
            sync_batchnorm=True,
            gradient_clip_val=3.0,
            precision="16-mixed" if not args.no_amp else "32-true",
            devices=args.gpus,
            accelerator="gpu",
            strategy="ddp_find_unused_parameters_true",
            # strategy="ddp",
            max_epochs=args.epochs,
            logger=wandb_logger if not args.debug else True,
            callbacks=[
                loss_checkpoint,
                lr_monitor,
                early_stopping,
            ],
            fast_dev_run=args.debug,
            num_sanity_val_steps=0,
            accumulate_grad_batches=16 // args.batch_size
            if args.batch_size < 16
            else 1,
            use_distributed_sampler=False,
            reload_dataloaders_every_n_epochs=1,
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
