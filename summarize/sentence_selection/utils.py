import numpy as np
import polars as pl
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def get_token_length(sentences):
    if len(sentences) == 0:
        return 0
    return [len(enc.encode(s)) for s in sentences]


def prefilter_sentences(df, regexes=None):
    """
    Use a regex to remove sentences that are found in tables, figures or supplementary material

    Optionally apply some user supplied regexes to remove sentences that match those regexes.
    """
    filtered_df = df.lazy().filter(
        pl.col("sentence")
        .arr.eval(
            ~pl.element().str.contains("image|figure|table|supplementary material")
        )
        .collect()
    )

    if regexes:
        for regex in regexes:
            filtered_df = (
                filtered_df.lazy()
                .filter(pl.col("sentence").str.contains(regex).is_not())
                .collect()
            )

    return filtered_df


def plot(clustered, outliers):
    import matplotlib.pyplot as plt
    import seaborn as sns

    ## Plot the clusters
    fig = plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=outliers.select("x").to_numpy().flatten(),
        y=outliers.select("y").to_numpy().flatten(),
        color="grey",
        alpha=0.4,
    )
    ax = sns.scatterplot(
        x=clustered.select("x").to_numpy().flatten(),
        y=clustered.select("y").to_numpy().flatten(),
        hue=clustered.select("sentence_labels").to_numpy().flatten(),
        legend="full",
        alpha=0.7,
        palette=sns.color_palette(
            "hls",
            len(np.unique(clustered.select("sentence_labels").to_numpy().flatten())),
        ),
    )
    handles, labels = ax.get_legend_handles_labels()
    print(clustered.get_column("topics").unique().to_numpy())
    lgd = ax.legend(
        handles,
        clustered.get_column("topics").unique().to_list(),
        bbox_to_anchor=(1, 1),
        title=None,
        frameon=False,
    )
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None, frameon=False)

    # fig.tight_layout()
    # plt.savefig("sentence_topics_model1.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
