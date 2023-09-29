import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### extract pipeline statistics (tree leaves).
from modules.benchmark_graph_utils import pipeline_tree, tree_plot
from modules.constants_settings import ConstantsSettings as CS
from typing import Dict, List


def benchmark_scatterplot(
    pipe_tree: pipeline_tree, x="coverage", y="precision", hue="recall"
):
    pipe_tree_df_leaves = pipe_tree.get_leaves_df()
    sns.scatterplot(
        data=pipe_tree_df_leaves,
        x=x,
        y=y,
        hue=hue,
        palette="mako",
        legend=True,
    )


def plot_benchmark_corrs(pip_tree: pipeline_tree):
    # plor palette: https://seaborn.pydata.org/tutorial/color_palettes.html
    # calculate the correlation matrix
    pipe_tree_df_leaves = pip_tree.get_leaves_df()
    corr = pipe_tree_df_leaves.corr()
    # plot the heatmap
    matrix = np.triu(corr)
    sns.heatmap(
        corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        mask=matrix,
    )
    plt.show()


def benchmark_results_pca(pipe_tree: pipeline_tree):
    pipe_tree_df_leaves = pipe_tree.get_leaves_df()

    X = np.array(pipe_tree_df_leaves)
    pca = PCA(n_components=4).fit(X)
    pipe_tree_df_pca = pd.DataFrame(
        pca.transform(X),
        columns=["pc1", "pc2", "pc3", "pc4"],
        index=pipe_tree_df_leaves.index,
    )

    pipe_tree_df_pca = pd.concat([pipe_tree_df_leaves, pipe_tree_df_pca], axis=1)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["pca1", "pca2", "pca3", "pca4"],
        index=pipe_tree_df_pca.columns[:-4],
    )

    return pipe_tree_df_pca, loadings


def plot_pipe_tree_pca(pipe_tree_df_pca):
    pipe_tree_df_pca["node"] = pipe_tree_df_pca.index
    fig = px.scatter(
        pipe_tree_df_pca, x="pc2", y="pc1", color="node", hover_data=["node"]
    )
    fig.update_traces(textposition="bottom right")

    fig.update_layout(
        autosize=False,
        width=1200,
        height=750,
    )

    fig.update_traces(
        marker=dict(
            size=12,
        ),
    )
    fig.show()

    fig = px.scatter(
        pipe_tree_df_pca, x="pc2", y="pc4", color="node", hover_data=["node"]
    )
    fig.update_traces(textposition="bottom right")

    fig.update_layout(
        autosize=False,
        width=1200,
        height=750,
    )

    fig.update_traces(
        marker=dict(
            size=12,
        ),
    )

    fig.show()


def plot_benchmark_tree(
    pipe_tree: pipeline_tree,
    internode_function=np.mean,
    stats=["precision", "coverage"],
):
    pipe_tree_df_leaves = pipe_tree.get_leaves_df()
    ###
    subset_stats = pipe_tree_df_leaves[stats]
    view_stats = pipe_tree_df_leaves.copy()
    view_stats["stat_combined"] = subset_stats.prod(axis=1)
    view_stats = view_stats.sort_values("stat_combined", ascending=False)

    ######## plot stat distribution

    fig = view_stats["stat_combined"].hist(figsize=(13, 2), bins=20)
    plt.show(fig)

    #########

    pipe_tree.tree_scores(stats, internode_function=internode_function)
    print("compressing tree")
    pipe_tree.compress_tree()

    nodes = pipe_tree.nodes_compress
    edges = pipe_tree.edge_compress
    weights = pipe_tree.weights

    ### tree plot
    pipe_graph = tree_plot()
    prox = pipe_tree.node_index.to_dict()["node"]
    prox = {k: [k, v] for k, v in prox.items()}
    pipe_graph.graph(nodes, prox, edges, weights)
    pipe_graph.generate_graph()
    graph_fig = pipe_graph.graph_plot()

    graph_fig.update_layout(
        autosize=False,
        width=1050,
        height=600,
    )

    graph_fig.show()


def plot_workflow_boxplots(
    pipe_tree: pipeline_tree,
    stats=["precision", "recall", "coverage"],
    select_modules_dict: Dict[str, List[str]] = None,
    technology: str = CS.TECHNOLOGY_minion,
):
    pipe_tree_df_pca, loadings = benchmark_results_pca(pipe_tree)
    ### boxplot analysis.

    stats = ["precision", "ahelp", "coverage"]
    stats = ["coverage", "ref_proportion", "depth"]
    stats = ["coverage", "precision"]
    stats = ["rhelp", "precision"]
    stats = [
        "precision",
    ]

    ##### selecting nodes by software
    leaf_paths = pipe_tree.get_leaf_paths()
    leaf_paths_index = {l: pipe_tree.node_index.loc[g] for l, g in leaf_paths.items()}

    stats_df = pipe_tree_df_pca.copy()

    def check_node_selected(row, select_modules_dict):
        node = row.name
        node_df = leaf_paths_index.get(node)
        final_name = []
        for m, soft_list in select_modules_dict.items():
            for s in soft_list:
                node = (m, s, "module")
                if node in list(node_df.node.values):
                    final_name.append(s)

        if len(final_name):
            return "+".join(final_name)
        else:
            return None

    stats_df["label"] = stats_df.apply(
        lambda x: check_node_selected(x, select_modules_dict), axis=1
    )
    stats_df = stats_df[stats_df["label"].notnull()]

    #### plot boxplot
    stats_df["stat_combined"] = stats_df[stats].prod(axis=1)

    plot_df_group_stats = stats_df[["stat_combined", "label"]].copy()
    plot_df_group_stats = (
        plot_df_group_stats.groupby("label")
        .median()
        .sort_values("stat_combined", ascending=False)
    )
    stats_df["median"] = stats_df["label"].map(
        plot_df_group_stats["stat_combined"].to_dict()
    )
    stats_df = stats_df.sort_values("median", ascending=False)

    ### sns plot

    fig = plt.figure(figsize=(10, 10))

    fig = sns.boxplot(data=stats_df, y="label", x="stat_combined", palette="Set3")
    fig.set_title(f"tech: {technology} stat:" + " * ".join(stats))

    # x label
    xlbal = "+".join(select_modules_dict.keys())

    fig.set_xlabel(f"modules: {xlbal}")

    plt.show(fig)


import matplotlib.pyplot as plt
import seaborn as sns


def plot_pipe_tree_heatmap(pipe_tree_df_leaves, columns_to_plot=[]):
    """
    plot pipe tree heatmap using viridis without values using sns (for paper)
    """
    df_to_plot = pipe_tree_df_leaves[columns_to_plot]

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 10)
    #
    sns.heatmap(df_to_plot)
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


def label_nodes(row, module, leaf_paths_index: dict) -> str:
    node = row.name
    node_df = leaf_paths_index.get(node)

    for node in list(node_df.node.values):
        if node[0] == module:
            return node[1]

    return "None"


###


def strings_to_factors(df, columns: List[str]):
    total_software = []
    for col in columns:
        total_software += list(df[col].unique())
    total_software = list(set(total_software))
    col_codes = {s: i for i, s in enumerate(total_software)}

    for col in columns:
        df[col] = df[col].apply(lambda x: col_codes.get(x))

    return df, col_codes


from modules.analysis_functions import (
    strings_to_factors,
    label_nodes,
    benchmark_results_pca,
)


def plot2heatmaps(pipe_tree: pipeline_tree, cols_categorical=[], cols_numerical=[]):
    leaf_paths = pipe_tree.get_leaf_paths()
    leaf_paths_index = {l: pipe_tree.node_index.loc[g] for l, g in leaf_paths.items()}

    pipe_tree_df_pca, loadings = benchmark_results_pca(pipe_tree)

    stats_df = pipe_tree_df_pca.copy()

    for col in cols_categorical:
        stats_df[col] = stats_df.apply(
            lambda x: label_nodes(x, col, leaf_paths_index), axis=1
        )

    stats_df_categorical, col_codes = strings_to_factors(stats_df, cols_categorical)

    plot_pipe_tree_heatmap_2plots(
        stats_df_categorical,
        cols_categorical=cols_categorical,
        cols_numerical=cols_numerical,
    )


def plot_pipe_tree_heatmap_2plots(
    stats_df: pd.DataFrame, cols_categorical=[], cols_numerical=[]
):
    """
    plot pipe tree heatmap using viridis without values using sns (for paper)
    """
    # pipe_tree_df_leaves = pipe_tree.get_leaves_df()
    df_categorical = stats_df[cols_categorical]
    cols_numerical = stats_df[cols_numerical]

    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")

    plt.rcParams["figure.figsize"] = (15, 10)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    with sns.color_palette("Set2") as cmap:
        sns.heatmap(df_categorical, cmap=cmap, ax=ax1)

    with sns.color_palette("viridis") as cmap:
        sns.heatmap(cols_numerical, cmap=cmap, ax=ax2)

    plt.show()
