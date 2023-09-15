### gather_runid_f1
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns


def filter_raw_ref(raw_refs, node_stats: list = []):
    raw_refs_analyse = raw_refs.copy()

    if len(node_stats):
        runs = node_runs_df[node_runs_df.node.isin(node_stats)]
        runs = [x for x in runs.runids.unique()]
        runs = [x.split("_")[-1] for x in runs]
        runs = [int(x) for x in runs]

        raw_refs_analyse = raw_refs_analyse[raw_refs_analyse.run_id.isin(runs)]

    return raw_refs_analyse


def sort_by_counts_simple(df: pd.DataFrame):
    counts = df.counts.values
    for ix, ct in enumerate(counts):
        if "/" in ct:
            counts[ix] = int(ct.split("/")[0])

    df["simple_counts"] = [float(x) for x in counts]
    df = df.sort_values("simple_counts", ascending=False)
    df.drop("simple_counts", axis=1, inplace=True)
    return df


def sort_by_counts_combined(df):
    df = df.sort_values("id", ascending=False)
    return df


def remap_threshold_stats(
    raw_refs_analyse: pd.DataFrame,
    max_remap: int = 20,
    sort_type: str = "simple_counts",
):
    f1_df = []
    precision_df = []
    recall_df = []

    for rid in raw_refs_analyse.run_id.unique():
        ridf = raw_refs_analyse[raw_refs_analyse.run_id == rid]

        if sort_type == "simple_counts":
            ridf = sort_by_counts_simple(ridf)
        elif sort_type == "combined":
            ridf = sort_by_counts_combined(ridf)

        f1_list = [rid]
        precision_list = [rid]
        recall_list = [rid]
        all_found = ridf.found.sum()

        for rmap in range(0, max_remap):
            rmap_sel = ridf.head(rmap)
            true_positives = rmap_sel.found.sum()
            false_positives = rmap_sel.shape[0] - true_positives
            false_negatives = all_found - true_positives

            # print(false_negatives)

            f1 = true_positives / (
                true_positives * 0.5 * (false_positives + false_negatives)
            )
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)

            f1_list.append(f1)
            precision_list.append(precision)
            recall_list.append(recall)

        f1_df.append(f1_list)
        precision_df.append(precision_list)
        recall_df.append(recall_list)

    f1_df = pd.DataFrame(
        f1_df, columns=["run_id"] + [f"rem_{r}" for r in range(0, max_remap)]
    )
    precision_df = pd.DataFrame(
        precision_df, columns=["run_id"] + [f"rem_{r}" for r in range(0, max_remap)]
    )
    recall_df = pd.DataFrame(
        recall_df, columns=["run_id"] + [f"rem_{r}" for r in range(0, max_remap)]
    )

    return f1_df, precision_df, recall_df


def clean_df_standardize(df: pd.DataFrame):
    df_new = df.copy()
    df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    intermediate = df_new.iloc[:, 1:].T / df_new.iloc[:, 1:].max(axis=1, skipna=True)
    df_new.iloc[:, 1:] = intermediate.T

    return df_new


import matplotlib.pyplot as plt
import numpy as np


def plot_sorting_benchmark(f1_df_play, precision_df_play, recall_df_play, max_remap=20):
    means_f1 = f1_df_play.iloc[:, 1:].mean(axis=0, skipna=True)
    std_f1 = f1_df_play.iloc[:, 1:].std(axis=0, skipna=True)
    f1_stats = pd.DataFrame(
        {
            "range": [r for r in range(0, max_remap)],
            "mean": means_f1,
            "std": std_f1,
            "stat": "f1",
        }
    )

    means_precision = precision_df_play.iloc[:, 1:].mean(axis=0, skipna=True)
    std_precision = precision_df_play.iloc[:, 1:].std(axis=0, skipna=True)
    precision_stats = pd.DataFrame(
        {
            "range": [r for r in range(0, max_remap)],
            "mean": means_precision,
            "std": std_precision,
            "stat": "precision",
        }
    )

    means_recall = recall_df_play.iloc[:, 1:].mean(axis=0, skipna=True)
    std_recall = recall_df_play.iloc[:, 1:].std(axis=0, skipna=True)
    recall_stats = pd.DataFrame(
        {
            "range": [r for r in range(0, max_remap)],
            "mean": means_recall,
            "std": std_recall,
            "stat": "recall",
        }
    )

    compound_stats = pd.concat(
        [f1_stats, precision_stats, recall_stats], axis=0
    ).reset_index(drop=True)
    compound_horizontal = pd.DataFrame(
        {
            "precision": means_precision,
            "recall": means_recall,
        }
    )

    ## plotly lineplot

    import plotly.express as px

    plt.figure(figsize=(10, 5))
    fig = px.line(compound_stats, x="range", y="mean", color="stat")

    # plotly image size

    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        paper_bgcolor="LightSteelBlue",
    )

    # Plotly xticks

    fig.update_xaxes(
        tickmode="array",
        tickvals=tuple(range(0, max_remap)),
    )

    fig.show()

    ## seaborn plot with std

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))

    sns.lineplot(data=compound_stats, x="range", y="mean", hue="stat")

    plt.xticks(range(0, max_remap))

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.errorbar(range(0, max_remap), means_f1, yerr=std_f1, fmt="o")
    plt.errorbar(range(0, max_remap), means_precision, yerr=std_precision, fmt="o")
    plt.errorbar(range(0, max_remap), means_recall, yerr=std_recall, fmt="o")
    plt.xticks(range(0, max_remap))
    plt.xlabel("TAXID threshold")
    plt.ylabel("mean")
    plt.legend(["f1", "precision", "recall"])
    plt.show()
