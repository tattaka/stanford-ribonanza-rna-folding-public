import argparse
import datetime
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm.auto import tqdm, trange
from train import (
    RibonanzaLightningModel,
    train_id_to_bpp_paths,
    train_id_to_bpp_paths_contrafold,
)

SEQ_PATH = "../../input/stanford-ribonanza-rna-folding-converted/"

class RNA_Dataset_OOF(Dataset):
    def __init__(self, df, mask_only=False, **kwargs):
        self.seq_map = {"A": 0, "C": 1, "G": 2, "U": 3}
        self.structure_map = {".": 1, "(": 2, ")": 3}  # Add
        df["L"] = df.sequence.apply(len)
        self.Lmax = df["L"].max()
        self.df = df
        self.mask_only = mask_only

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_min, id_max, seq, structure, seq_id = self.df.loc[
            idx, ["id_min", "id_max", "sequence", "structure", "sequence_id"]
        ]
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        L = len(seq)
        mask[:L] = True
        if self.mask_only:
            return {"mask": mask}, {}
        ids = np.arange(id_min, id_max + 1)

        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        seq = np.pad(seq, (0, self.Lmax - L))
        bp_matrix = np.load(train_id_to_bpp_paths[seq_id])  # Add
        bp_matrix = np.pad(
            bp_matrix,
            ((0, self.Lmax - len(bp_matrix)), (0, self.Lmax - len(bp_matrix))),
        )  # Add
        bp_matrix_contrafold = np.load(train_id_to_bpp_paths_contrafold[seq_id])  # Add
        bp_matrix_contrafold = np.pad(
            bp_matrix_contrafold,
            ((0, self.Lmax - len(bp_matrix_contrafold)), (0, self.Lmax - len(bp_matrix_contrafold))),
        )  # Add
        structure = [self.structure_map[s] for s in structure]  # Add
        structure = np.array(structure)  # Add
        structure = np.pad(structure, (0, self.Lmax - len(structure)))  # Add
        ids = np.pad(ids, (0, self.Lmax - L), constant_values=-1)

        return {
            "seq": torch.from_numpy(seq),
            "mask": mask,
            "bp_matrix": torch.from_numpy(bp_matrix),  # Add
            "bp_matrix_contrafold": torch.from_numpy(bp_matrix_contrafold),
            "structure": torch.from_numpy(structure),  # Add
        }, {"ids": ids}


def dict_to(x, device="cuda"):
    return {k: x[k].to(device) for k in x}


def to_device(x, device="cuda"):
    return tuple(dict_to(e, device) for e in x)


class DeviceDataLoader:
    def __init__(self, dataloader, device="cuda"):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)


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
        "--batch_size",
        default=512,
        type=int,
        metavar="BS",
        help="batch_size",
        dest="batch_size",
    )
    dt_now = datetime.datetime.now()
    parent_parser.add_argument(
        "--logdir",
        default=f"{dt_now.strftime('%Y%m%d-%H-%M-%S')}",
    )
    parser = RibonanzaLightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(args.seed, workers=True)
    warnings.simplefilter("ignore")
    df_val = pd.read_parquet(
        os.path.join(SEQ_PATH, "val_sequences_structures_kfold.parquet")
    )
    ids, preds = [], []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for fold, (train_split, val_split) in enumerate(kf.split(np.arange(len(df_val)))):
        ds = RNA_Dataset_OOF(df_val.iloc[val_split].reset_index(drop=True))
        dl = DeviceDataLoader(
            torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=24,
            ),
            device,
        )
        resume_checkpoint = glob(
            f"../../logs/{args.logdir}/fold{fold}/**/best_loss.ckpt", recursive=True
        )[0]
        print(f"load checkpoint: {resume_checkpoint}")
        model = (
            RibonanzaLightningModel.load_from_checkpoint(resume_checkpoint)
            .model_ema.module.to(device)
            .eval()
        )
        logdir = f"../../logs/{args.logdir}/fold{fold}"
        print(f"logdir = {logdir}")
        for x, y in tqdm(dl):
            with torch.no_grad(), torch.cuda.amp.autocast():
                p = torch.nan_to_num(model(x))
            for idx, mask, pi in zip(y["ids"].cpu(), x["mask"].cpu(), p.cpu()):
                ids.append(idx[mask])
                preds.append(pi[mask[: pi.shape[0]]])

    ids = torch.concat(ids)
    preds = torch.concat(preds)

    df = pd.DataFrame(
        {
            "id": ids.numpy(),
            "reactivity_DMS_MaP": preds[:, 1].numpy().astype("float"),
            "reactivity_2A3_MaP": preds[:, 0].numpy().astype("float"),
        }
    )
    df = df.sort_values(by="id").reset_index(drop=True)
    df.to_parquet(
        f"../../logs/{args.logdir}/oof.parquet",
        index=False,
    )  
    print(df.head())


if __name__ == "__main__":
    main(get_args())
