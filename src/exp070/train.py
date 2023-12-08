import argparse
import datetime
import gc
import math
import os
import warnings
from collections import OrderedDict
from glob import glob
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info
from sklearn.model_selection import KFold
from timm.layers import RmsNorm
from timm.utils import ModelEmaV2
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
)
from transformers import get_cosine_schedule_with_warmup

EXP_ID = "070"
COMMENT = """
    kfold, postnorm, high weight decay, long warmup, low s/n threshold, 
    conv transformer, SHAPE positional encoding, bpps bias, efficient impl, 
    param tuning from exp034, swiGLU, split attention ALiBi and bpps, 
    fixed 0-1 clipping, B2T connection option, low grad clipping, add norm and act for conv1d
    with pseudo_label, RMSNorm
    """
SEQ_PATH = "../../input/stanford-ribonanza-rna-folding-converted/"
train_bpp_paths = glob(f"{SEQ_PATH}/bp_matrix/train*/*.npy", recursive=True)
train_id_to_bpp_paths = {p.split("/")[-1].split(".")[0]: p for p in train_bpp_paths}
test_bpp_paths = glob(f"{SEQ_PATH}/bp_matrix/test*/*.npy", recursive=True)
test_id_to_bpp_paths = {p.split("/")[-1].split(".")[0]: p for p in test_bpp_paths}
train_id_to_bpp_paths.update(test_id_to_bpp_paths)


def get_optimizerd_dtype(df):
    key_to_dtype = OrderedDict()
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            if str(col_type)[:3] == "int":
                key_to_dtype[col] = "int32"
            else:
                key_to_dtype[col] = "float16"
        else:
            key_to_dtype[col] = "category"
    return key_to_dtype


class RibonanzaDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: np.ndarray,
        mode: str = "train",
        mask_only: bool = False,
        filter_rate=0.5,
        **kwargs,
    ):
        self.mode = mode
        self.seq_map = {"A": 0, "C": 1, "G": 2, "U": 3}
        self.structure_map = {".": 1, "(": 2, ")": 3}  # Add
        df["L"] = df.sequence.apply(len)
        # self.Lmax = df["L"].max()
        self.Lmax = 457 if mode == "train" else df["L"].max()
        df_2A3 = df.loc[df.experiment_type == "2A3_MaP"]
        df_DMS = df.loc[df.experiment_type == "DMS_MaP"]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        if mode != "train":
            m = (df_2A3["SN_filter"].values > 0) & (df_DMS["SN_filter"].values > 0)
            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)
        else:
            m = (df_2A3["signal_to_noise"].values > filter_rate) & (
                df_DMS["signal_to_noise"].values > filter_rate
            )
            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)

        self.seq_id = df_2A3["sequence_id"].values  # Add
        self.seq = df_2A3["sequence"].values
        # self.structure = df_2A3["structure"].values  # Add
        self.L = df_2A3["L"].values

        self.react_2A3 = df_2A3[
            [c for c in df_2A3.columns if "reactivity_0" in c]
        ].values
        self.react_DMS = df_DMS[
            [c for c in df_DMS.columns if "reactivity_0" in c]
        ].values
        # self.react_err_2A3 = df_2A3[
        #     [c for c in df_2A3.columns if "reactivity_error_0" in c]
        # ].values
        # self.react_err_DMS = df_DMS[
        #     [c for c in df_DMS.columns if "reactivity_error_0" in c]
        # ].values
        self.sn_2A3 = df_2A3["signal_to_noise"].values
        self.sn_DMS = df_DMS["signal_to_noise"].values
        self.mask_only = mask_only
        self.data_length = len(self.seq)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx: int):
        seq_id = self.seq_id[idx]  # Add
        seq = self.seq[idx]
        # structure = self.structure[idx]  # Add
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

        # structure = [self.structure_map[s] for s in structure]  # Add
        # structure = np.array(structure)  # Add
        # structure = np.pad(structure, (0, self.Lmax - len(structure)))  # Add
        react_2A3 = self.react_2A3[idx]
        react_DMS = self.react_DMS[idx]
        # react_err_2A3 = self.react_err_2A3[idx]
        # react_err_DMS = self.react_err_DMS[idx]
        react = np.stack([react_2A3, react_DMS], -1)
        # react_err = np.stack([react_err_2A3, react_err_DMS], -1)
        react = np.pad(
            react, ((0, self.Lmax - len(react)), (0, 0)), constant_values=np.NaN
        )
        # react_err = np.pad(
        #     react_err, ((0, self.Lmax - len(react_err)), (0, 0)), constant_values=np.NaN
        # )
        sn = torch.FloatTensor([self.sn_2A3[idx], self.sn_DMS[idx]])
        return {
            "seq": torch.from_numpy(seq),
            "mask": mask,
            "bp_matrix": torch.from_numpy(bp_matrix),  # Add
            # "structure": torch.from_numpy(structure),  # Add
        }, {
            "react": torch.from_numpy(react),
            # "react_err": torch.from_numpy(react_err),
            "sn": sn,
            "mask": mask,
        }


class LenMatchBatchSampler(BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.dataset[idx]
            if isinstance(s, tuple):
                L = s[0]["mask"].sum()
            else:
                L = s["mask"].sum()
            L = max(1, L // 16)
            if len(buckets[L]) == 0:
                buckets[L] = []
            buckets[L].append(idx)

            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch


def dict_to(x, device="cuda"):
    return {k: x[k].to(device) for k in x}


def to_device(x, device="cuda"):
    return tuple(dict_to(e, device) for e in x)


class RibonanzaDataModule(LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        pseudo_label_df: pd.DataFrame,
        train_split: np.ndarray,
        val_split: np.ndarray,
        num_workers: int = 4,
        batch_size: int = 16,
        seed: int = 0,
    ):
        super().__init__()

        self._num_workers = num_workers
        self._batch_size = batch_size
        self.df = df
        self.pseudo_label_df = pseudo_label_df
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        self.save_hyperparameters(
            "num_workers",
            "batch_size",
        )

    def create_dataset(self, mode: str = "train") -> RibonanzaDataset:
        if mode == "train":
            train_dataset = RibonanzaDataset(
                df=self.df,
                split=self.train_split,
                mode=mode,
                mask_only=False,
            )
            train_dataset_mask = RibonanzaDataset(
                df=self.df,
                split=self.train_split,
                mode=mode,
                mask_only=True,
            )
            pl_split = np.random.permutation(len(self.pseudo_label_df) // 2)[
                : len(train_dataset)
            ]
            pl_dataset = RibonanzaDataset(
                df=self.pseudo_label_df,
                split=pl_split,
                mode=mode,
                mask_only=False,
                filter_rate=0.75,
            )
            pl_dataset_mask = RibonanzaDataset(
                df=self.pseudo_label_df,
                split=pl_split,
                mode=mode,
                mask_only=True,
                filter_rate=0.75,
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

    def __dataloader(self, mode: str = "train") -> DataLoader:
        """Train/validation loaders."""
        dataset, dataset_len = self.create_dataset(mode)
        subsampler = DistributedSampler(
            dataset_len, shuffle=(mode == "train"), seed=self.seed
        )
        sampler = LenMatchBatchSampler(
            subsampler,
            batch_size=self._batch_size,
            drop_last=(mode == "train"),
        )
        return DataLoader(
            dataset=dataset,
            num_workers=self._num_workers,
            pin_memory=False,
            batch_sampler=sampler,
            prefetch_factor=10,
        )

    def train_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="train")

    def val_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="valid")

    def test_dataloader(self) -> DataLoader:
        return self.__dataloader(mode="test")

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("RibonanzaDataModule")
        parser.add_argument(
            "--num_workers",
            default=6,
            type=int,
            metavar="W",
            help="number of CPU workers",
            dest="num_workers",
        )
        parser.add_argument(
            "--batch_size",
            default=64,
            type=int,
            metavar="BS",
            help="number of sample in a batch",
            dest="batch_size",
        )
        return parent_parser


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202
# https://github.com/lucidrains/PaLM-pytorch/blob/main/palm_pytorch/palm_pytorch.py#L57-L64


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# https://github.com/jaketae/alibi/blob/main/alibi/attention.py#L10-L22
def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    pos = x - y
    return (pos > 0) * -pos + (pos < 0) * pos


# https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742-L752
def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


class BiasedConvTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        kernel_size,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        b2t_connection: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.b2t_connection = b2t_connection
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=kernel_size)
        self.conv1d_t = nn.ConvTranspose1d(d_model, d_model, kernel_size=kernel_size)
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=kernel_size),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.conv2d_t = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=kernel_size),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward * 2)  # for swiGLU
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = RmsNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RmsNorm(d_model, eps=layer_norm_eps)
        self.norm3 = RmsNorm(d_model, eps=layer_norm_eps)
        self.norm4 = RmsNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        src: torch.Tensor,
        bias: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x_att, bias_att = self._sa_block(
                self.norm1(x), bias, src_mask, src_key_padding_mask
            )
            x = x + x_att
            x = x + self._ff_block(self.norm2(x))
            bias = bias + bias_att
        else:
            x_att, bias_att = self._sa_block(x, bias, src_mask, src_key_padding_mask)
            x = self.norm1(x + x_att)
            if self.b2t_connection:
                x = self.norm2(x + self._ff_block(x) + src)
            else:
                x = self.norm2(x + self._ff_block(x))
            bias = bias + bias_att
        return x, bias

    def _sa_block(
        self,
        x: torch.Tensor,
        bias: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.norm3(self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1))
        bias = self.conv2d(bias)
        pos = (
            get_relative_positions(bias.shape[-1])
            .to(dtype=bias.dtype, device=bias.device)
            .repeat(bias.shape[0], self.self_attn.num_heads // 2, 1, 1)
        )
        m = torch.tensor(
            get_slopes(self.self_attn.num_heads // 2),
            dtype=bias.dtype,
            device=bias.device,
        )[None, :, None, None]
        attn_mask, key_padding_mask = self._resize_mask(attn_mask, key_padding_mask, x)
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=x.dtype,
        )
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=torch.cat(
                [bias.repeat(1, self.self_attn.num_heads // 2, 1, 1), pos * m], 1
            ).reshape(-1, bias.shape[-2], bias.shape[-1]),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = F.gelu(self.norm4(self.conv1d_t(x.permute(0, 2, 1)).permute(0, 2, 1)))
        bias = self.conv2d_t(bias)
        return self.dropout1(x), bias

    def _resize_mask(self, src_mask, src_key_padding_mask, x):
        src_key_padding_mask = (
            F.interpolate(
                src_key_padding_mask[:, None].to(dtype=x.dtype),
                x.shape[1],
            )[:, 0]
            > 0.5
        )
        if src_mask is not None:
            src_mask = (
                F.interpolate(
                    src_mask[:, None].to(dtype=x.dtype),
                    (x.shape[1], x.shape[1]),
                )[:, 0]
                > 0.5
            )
        return src_mask, src_key_padding_mask


class RibonanzaModel(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 7,
        b2t_connection: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.seq_emb = nn.Embedding(4, dim)
        transformer_encoder = []
        for i in range(depth):
            transformer_encoder.append(
                BiasedConvTransformerEncoderLayer(
                    kernel_size=kernel_size if i < depth - 1 else 1,
                    d_model=dim,
                    nhead=dim // head_size,
                    dim_feedforward=4 * dim,
                    dropout=0.1,
                    activation=SwiGLU(),
                    norm_first=False,
                    b2t_connection=b2t_connection,
                )
            )
        self.transformer_encoder = nn.ModuleList(transformer_encoder)
        self.proj_out = nn.Linear(dim, 2)

    def forward(self, x0):
        mask = x0["mask"]
        Lmax = mask.sum(-1).max()
        mask = mask[:, :Lmax]
        x_seq = x0["seq"][:, :Lmax]
        bias_bpps = x0["bp_matrix"][:, None, :Lmax, :Lmax]
        x = self.seq_emb(x_seq)
        for i in range(len(self.transformer_encoder)):
            x, bias_bpps = self.transformer_encoder[i](
                x, bias_bpps, src_key_padding_mask=~mask
            )
        x = self.proj_out(x)
        return x


class RibonanzaLightningModel(pl.LightningModule):
    def __init__(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 7,
        b2t_connection: bool = False,
        lr: float = 1e-3,
        disable_compile: bool = False,
        no_amp: bool = False,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.no_amp = no_amp
        self.__build_model(
            dim=dim,
            depth=depth,
            head_size=head_size,
            kernel_size=kernel_size,
            b2t_connection=b2t_connection,
        )
        if not disable_compile:
            self.__compile_model()
        self.save_hyperparameters()

    def __build_model(
        self,
        dim: int = 192,
        depth: int = 12,
        head_size: int = 32,
        kernel_size: int = 7,
        b2t_connection: bool = False,
    ):
        self.model = RibonanzaModel(
            dim=dim,
            depth=depth,
            head_size=head_size,
            kernel_size=kernel_size,
            b2t_connection=b2t_connection,
        )
        self.model_ema = ModelEmaV2(self.model, decay=0.999)
        self.criterions = {"l1": nn.L1Loss(reduction="none")}

    def __compile_model(self):
        self.model = torch.compile(self.model)
        self.model_ema = torch.compile(self.model_ema)

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        losses = {}
        preds = outputs["preds"]
        targets = labels["targets"]
        p = preds[targets["mask"][:, : preds.shape[1]]]
        y = targets["react"][targets["mask"]]
        l1_loss = self.criterions["l1"](p, y)
        if self.training:
            l1_loss = torch.where(
                torch.logical_or(
                    torch.logical_and(p > 10, y > 10),
                    torch.logical_and(p < -10, y < -10),
                ),
                0,
                l1_loss,
            )
        l1_loss = l1_loss[~torch.isnan(l1_loss)].mean()
        losses["loss"] = l1_loss
        losses["l1_loss"] = l1_loss
        return losses

    def training_step(self, batch, batch_idx):
        self.model_ema.update(self.model)
        step_output = {}
        outputs = {}
        loss_target = {}
        input, label = batch
        outputs["preds"] = self.model(input)
        loss_target["targets"] = label
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.log_dict(
            dict(
                train_loss=losses["loss"],
                train_l1_loss=losses["l1_loss"],
            )
        )
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = {}
        outputs = {}
        loss_target = {}

        input, label = batch
        outputs["preds"] = self.model_ema.module(input).clip(0, 1)
        loss_target["targets"] = label
        loss_target["targets"]["react"][loss_target["targets"]["mask"]] = loss_target[
            "targets"
        ]["react"][loss_target["targets"]["mask"]].clip(0, 1)
        losses = self.calc_loss(outputs, loss_target)
        step_output.update(losses)
        self.log_dict(
            dict(
                val_loss=losses["loss"],
                val_l1_loss=losses["l1_loss"],
            )
        )
        return step_output

    def get_optimizer_parameters(self):
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0,
                "lr": self.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.05,
                "lr": self.lr,
            },
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        self.warmup = True
        optimizer = AdamW(
            self.get_optimizer_parameters(), eps=1e-6 if not self.no_amp else 1e-8
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

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = parent_parser.add_argument_group("RibonanzaLightningModel")
        parser.add_argument(
            "--dim",
            default=192,
            type=int,
            metavar="D",
            dest="dim",
        )
        parser.add_argument(
            "--depth",
            default=12,
            type=int,
            metavar="DPT",
            dest="depth",
        )
        parser.add_argument(
            "--head_size",
            default=32,
            type=int,
            metavar="HS",
            dest="head_size",
        )
        parser.add_argument(
            "--kernel_size",
            default=7,
            type=int,
            metavar="KM",
            dest="kernel_size",
        )
        parser.add_argument(
            "--b2t_connection",
            action="store_true",
            help="b2t_connection option",
            dest="b2t_connection",
        )
        parser.add_argument(
            "--lr",
            default=5e-4,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--disable_compile",
            action="store_true",
            help="disable torch.compile",
            dest="disable_compile",
        )
        return parent_parser


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
    parser = RibonanzaLightningModel.add_model_specific_args(parent_parser)
    parser = RibonanzaDataModule.add_model_specific_args(parser)
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
        datamodule = RibonanzaDataModule(
            df=df,
            pseudo_label_df=pseudo_label_df,
            train_split=train_split,
            val_split=val_split,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        model = RibonanzaLightningModel(
            dim=args.dim,
            depth=args.depth,
            head_size=args.head_size,
            kernel_size=args.kernel_size,
            b2t_connection=args.b2t_connection,
            lr=args.lr,
            disable_compile=args.disable_compile,
            no_amp=args.no_amp,
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
            monitor="val_loss", patience=10, log_rank_zero_only=True
        )
        os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)
        if not args.debug:
            wandb_logger = WandbLogger(
                name=f"exp{EXP_ID}/{args.logdir}/fold{fold}",
                save_dir=logdir,
                project="stanford-ribonanza-rna-folding",
                tags=["full_data"],
            )

        trainer = pl.Trainer(
            default_root_dir=logdir,
            sync_batchnorm=True,
            gradient_clip_val=1.0,
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
            accumulate_grad_batches=64 // args.batch_size
            if args.batch_size < 64
            else 1,
            use_distributed_sampler=False,
            reload_dataloaders_every_n_epochs=1,
            max_time={"days": 2, "hours": 12},
        )
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main(get_args())
