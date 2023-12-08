import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


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
        "--logdir",
        default=f"",
    )
    parent_parser.add_argument(
        "--fold",
        type=int,
        default=0,
    )
    parent_parser.add_argument(
        "--filter",
        type=float,
        default=0.5,
    )
    return parent_parser.parse_args()


def main(args):
    logdir = f"../../logs/{args.logdir}/fold{args.fold}/"
    filename = logdir + "test_pseudo_label"
    pl_path = sorted(glob(f"{filename}_*"))
    df_filter_2A3 = []
    df_filter_DMS = []
    df_all_len = 0
    filtered_len = 0
    for p in tqdm(pl_path):
        df = pd.read_parquet(p)
        df_2A3 = df.loc[
            (df.experiment_type == "2A3_MaP") & (df.future == 1)
        ].reset_index(drop=True)
        df_DMS = df.loc[
            (df.experiment_type == "DMS_MaP") & (df.future == 1)
        ].reset_index(drop=True)
        m = ((df_2A3["signal_to_noise"].values > args.filter)) & (
            (df_DMS["signal_to_noise"].values > args.filter)
        )
        df_filter_2A3.append(df_2A3.loc[m])
        df_filter_DMS.append(df_DMS.loc[m])
        df_all_len += len(df_2A3)
        filtered_len += m.sum()
    new_df = pd.concat(df_filter_2A3 + df_filter_DMS).reset_index(drop=True)
    new_df = new_df.drop(["id_min", "id_max", "structure", "future"], axis=1)
    start_mem = new_df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    reduce_keys = [key for key in new_df if "reactivity" in key] + [
        "signal_to_noise",
        "free_energy",
    ]
    new_df[reduce_keys] = new_df[reduce_keys].astype("float16")
    end_mem = new_df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    new_name = f"test_pl_filterling_{args.filter}_half.csv"
    new_df.to_csv(
        f"{os.path.join(logdir, new_name)}",
        index=False,
    )
    print(f"filterd length: {filtered_len}/{df_all_len} using {args.filter}")


if __name__ == "__main__":
    main(get_args())
