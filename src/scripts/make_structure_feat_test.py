import os

os.environ["ETERNAFOLD_PATH"] = "/opt/conda/bin/eternafold-bin"
os.environ[
    "ETERNAFOLD_PARAMETERS"
] = "/opt/conda/lib/eternafold-lib/parameters/EternaFoldParams.v1"

import warnings

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

warnings.filterwarnings("ignore")


import time
from multiprocessing import Pool

import arnie.utils as utils
import numpy as np
import pandas as pd
from arnie.bpps import bpps
from arnie.free_energy import free_energy
from arnie.mea.mea import MEA
from arnie.mfe import mfe
from arnie.pfunc import pfunc
from tqdm.auto import tqdm


def proc1(arg):
    sequence = arg[0]
    id = arg[1]
    log_gamma = arg[2]
    bp_matrix = bpps(sequence, package="eternafold")
    mea_mdl = MEA(bp_matrix, gamma=10**log_gamma)
    energy = free_energy(sequence, mea_mdl.structure, package="eternafold")
    return (
        id,
        sequence,
        mea_mdl.structure,
        log_gamma,
        mea_mdl.score_expected()[2],
        energy,
        bp_matrix,
    )


MAX_THRE = 40
CHUNK_NUM = 10000
test_data = pd.read_parquet(
    "../../input/stanford-ribonanza-rna-folding-converted/test_sequences.parquet"
)
chunk_start = 0
new_df = pd.DataFrame()
bp_matrix_dir = "../../input/stanford-ribonanza-rna-folding-converted/bp_matrix"
p = Pool(processes=MAX_THRE)
while chunk_start < (len(test_data)):
    os.makedirs(
        os.path.join(bp_matrix_dir, f"test_{chunk_start}_{chunk_start + CHUNK_NUM}"),
        exist_ok=True,
    )
    print(
        f"Start process: {chunk_start} - {chunk_start + CHUNK_NUM} / {len(test_data)}"
    )
    test_data_small = test_data.iloc[chunk_start : chunk_start + CHUNK_NUM]
    target_df = test_data_small[["sequence_id", "sequence"]]
    total_items = len(target_df)
    n_gamma = 5
    li = []
    for log_gamma in range(n_gamma):
        for i, arr in enumerate(target_df[["sequence", "sequence_id"]].values):
            li.append([arr[0], arr[1], log_gamma])
    results = []
    start_time = time.time()  # Start tracking time
    for i, ret in enumerate(p.imap(proc1, li)):
        results.append(ret)
        elapsed_time = time.time() - start_time
        estimated_total_time = (elapsed_time / (i + 1)) * n_gamma * total_items
        remaining_time = estimated_total_time - elapsed_time
        print(
            f"\rProcessed item {i + 1}/{n_gamma*total_items}. Elapsed time: {elapsed_time:.2f}s. Estimated remaining time: {remaining_time:.2f}s.",
            end="",
        )
    df = pd.DataFrame(
        results,
        columns=[
            "sequence_id",
            "sequence",
            "structure",
            "log_gamma",
            "score",
            "free_energy",
            "bp_matrix",
        ],
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    for sequence_id in tqdm(target_df["sequence_id"].unique()):
        unq_df = df[df["sequence_id"] == sequence_id].drop_duplicates("structure")
        unq_df.free_energy = unq_df.free_energy.astype(np.float32)
        np.save(
            os.path.join(
                bp_matrix_dir,
                f"test_{chunk_start}_{chunk_start + CHUNK_NUM}",
                sequence_id,
            ),
            unq_df.loc[:, ["bp_matrix"]].iloc[0].to_numpy()[0].astype(np.float32),
        )
        new_df = pd.concat(
            [
                new_df,
                unq_df.loc[
                    :,
                    [
                        "sequence_id",
                        "sequence",
                        "structure",
                        "log_gamma",
                        "free_energy",
                    ],
                ].iloc[:1],
            ]
        )
    chunk_start += CHUNK_NUM
new_df.reset_index(drop=True)
new_df.to_parquet(
    "../../input/stanford-ribonanza-rna-folding-converted/test_structure.parquet",
    index=False,
)
