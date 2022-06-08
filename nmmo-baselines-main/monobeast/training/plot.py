import pandas as pd
from matplotlib import pyplot as plt
import argparse

STATS = [
    "total_loss", "mean_episode_return", "mean_episode_step", "pg_loss",
    "baseline_loss", "entropy_loss", "rho", "advantage", "grad_norm"
]
N_COLUMN = 3


def plot(log_file, window=500):
    log_df = pd.read_csv(log_file, sep=",")
    for key in STATS + ["step"]:
        assert key in log_df.columns
    fig, axes = plt.subplots(len(STATS) // N_COLUMN, N_COLUMN)
    fig.suptitle("Learning curves on IJCAI2022-NMMO PVE STAGE1", fontsize=20)
    fig.set_size_inches(18, 10)
    for i, key in enumerate(STATS):
        df = log_df[["step", key]].dropna()
        # df = df.groupby(lambda x: x // average_window).mean()
        df = df.rolling(window).mean()
        ax = axes[i // N_COLUMN, i % N_COLUMN]
        ax.ticklabel_format(style='sci', scilimits=(-1, 4), axis='x')
        ax.set_title(key)
        ax.plot(df["step"], df[key])
    plt.savefig("plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logfile",
        type=str,
        default="results/nmmo/logs.csv",
    )
    args = parser.parse_args()
    plot(args.logfile)
