import argparse
import datetime
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
from torch.utils.data import Dataset
from train import (
    EXP_ID,
    SEQ_PATH,
    RibonanzaDataModule,
    RibonanzaLightningModel,
    train_id_to_bpp_paths,
)
from transformers import get_cosine_schedule_with_warmup

EXP_ID = f"{EXP_ID}_finetune"


class RibonanzaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: np.ndarray,
        mode: str = "train",
        mask_only: bool = False,
        **kwargs,
    ):
        self.mode = mode
        self.seq_map = {"A": 0, "C": 1, "G": 2, "U": 3}
        self.structure_map = {".": 1, "(": 2, ")": 3}  # Add
        df["L"] = df.sequence.apply(len)
        self.Lmax = df["L"].max()
        df_2A3 = df.loc[df.experiment_type == "2A3_MaP"]
        df_DMS = df.loc[df.experiment_type == "DMS_MaP"]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        m = (df_2A3["SN_filter"].values > 0) & (df_DMS["SN_filter"].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq_id = df_2A3["sequence_id"].values  # Add
        self.seq = df_2A3["sequence"].values
        self.structure = df_2A3["structure"].values  # Add
        self.L = df_2A3["L"].values

        self.react_2A3 = df_2A3[
            [c for c in df_2A3.columns if "reactivity_0" in c]
        ].values
        self.react_DMS = df_DMS[
            [c for c in df_DMS.columns if "reactivity_0" in c]
        ].values
        self.react_err_2A3 = df_2A3[
            [c for c in df_2A3.columns if "reactivity_error_0" in c]
        ].values
        self.react_err_DMS = df_DMS[
            [c for c in df_DMS.columns if "reactivity_error_0" in c]
        ].values
        self.sn_2A3 = df_2A3["signal_to_noise"].values
        self.sn_DMS = df_DMS["signal_to_noise"].values
        self.mask_only = mask_only

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx: int):
        seq_id = self.seq_id[idx]  # Add
        seq = self.seq[idx]
        structure = self.structure[idx]  # Add
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[: len(seq)] = True
            return {"mask": mask}, {"mask": mask}
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[: len(seq)] = True
        seq = np.pad(seq, (0, self.Lmax - len(seq)))
        bp_matrix = np.load(train_id_to_bpp_paths[seq_id])  # Add
        bp_matrix = np.pad(
            bp_matrix,
            ((0, self.Lmax - len(bp_matrix)), (0, self.Lmax - len(bp_matrix))),
        )  # Add
        structure = [self.structure_map[s] for s in structure]  # Add
        structure = np.array(structure)  # Add
        structure = np.pad(structure, (0, self.Lmax - len(structure)))  # Add
        react_2A3 = self.react_2A3[idx]
        react_DMS = self.react_DMS[idx]
        react_err_2A3 = self.react_err_2A3[idx]
        react_err_DMS = self.react_err_DMS[idx]
        react = np.stack([react_2A3, react_DMS], -1)
        react_err = np.stack([react_err_2A3, react_err_DMS], -1)

        sn = torch.FloatTensor([self.sn_2A3[idx], self.sn_DMS[idx]])

        return {
            "seq": torch.from_numpy(seq),
            "mask": mask,
            "bp_matrix": torch.from_numpy(bp_matrix),  # Add
            "structure": torch.from_numpy(structure),  # Add
        }, {
            "react": torch.from_numpy(react),
            "react_err": torch.from_numpy(react_err),
            "sn": sn,
            "mask": mask,
        }


class RibonanzaDataFTModule(RibonanzaDataModule):
    def create_dataset(self, mode: str = "train") -> RibonanzaDataset:
        if mode == "train":
            return (
                RibonanzaDataset(
                    df=self.df,
                    split=self.train_split,
                    mode=mode,
                    mask_only=False,
                ),
                RibonanzaDataset(
                    df=self.df,
                    split=self.train_split,
                    mode=mode,
                    mask_only=True,
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
    df = pd.read_parquet(
        os.path.join(SEQ_PATH, "train_data_kmeans_groupkfold_structures.parquet")
    )
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    assert args.fold < 5
    for fold, (train_split, val_split) in enumerate(kf.split(np.arange(len(df) // 2))):
        if args.fold != fold:
            continue
        datamodule = RibonanzaDataFTModule(
            df=df,
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
            monitor="val_l1_loss",
            save_top_k=1,
            save_last=True,
            mode="min",
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_l1_loss", log_rank_zero_only=True
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
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
