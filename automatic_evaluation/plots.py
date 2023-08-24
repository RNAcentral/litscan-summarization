from pathlib import Path

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import obonet
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import cohen_kappa_score


def anonymise_data(data):
    data = data.with_columns(
        pl.col("user_id").apply(lambda x: "Nancy" if x == "anonymous" else x)
    )

    data = data.with_columns(
        pl.when(pl.col("user_id").eq("Nancy"))
        .then("Rater_0")
        .when(pl.col("user_id").eq("blake"))
        .then("Rater_1")
        .when(pl.col("user_id").eq("andrew"))
        .then("Rater_2")
        .when(pl.col("user_id").eq("Alex"))
        .then("Rater_3")
        .alias("user_id")
    )

    return data


def scatterFilter(x, y, **kwargs):
    r1 = x.name
    r2 = y.name
    interimDf = pd.concat([x, y, kwargs["rna_ids"]], axis=1)
    interimDf.columns = ["x", "y", "id"]
    interimDf = pl.from_pandas(interimDf)
    kwargs = {}
    interimDf = (
        interimDf.groupby("id").agg([pl.col("x").max(), pl.col("y").max()]).drop_nulls()
    )

    x = interimDf.get_column("x").to_numpy()
    y = interimDf.get_column("y").to_numpy()

    K = cohen_kappa_score(x, y)

    ax = plt.gca()
    ax.plot(
        x + np.random.normal(0, 0.1, interimDf.height),
        y + np.random.normal(0, 0.1, interimDf.height),
        ".",
        **kwargs,
    )
    ax.set_xlim([0, 6])
    ax.set_ylim([0, 6])
    ax.text(1, 5.5, "$\kappa$ = " + str(round(K, 2)))


@click.command()
@click.option(
    "--output_path",
    default="/Users/agreen/Dropbox/Work/ARISE/Projects/LitSumm/figures/",
    type=click.Path(),
)
@click.option(
    "--feedback_pair_grid",
    is_flag=True,
    default=False,
    help="Plot a pair grid of feedback data",
)
@click.option(
    "--feedback_box_plot",
    is_flag=True,
    default=False,
    help="Plot a box plot of feedback data",
)
@click.option(
    "--feedback_pair_plot",
    is_flag=True,
    default=False,
    help="Plot a pair plot of feedback data",
)
@click.option(
    "--rouge_kdeplot",
    is_flag=True,
    default=False,
    help="Plot a KDE plot of ROUGE scores",
)
@click.option(
    "--rouge_kdeplot_fb",
    is_flag=True,
    default=False,
    help="Plot a KDE plot of ROUGE scores for feedback data",
)
@click.option(
    "--rouge_histplot",
    is_flag=True,
    default=False,
    help="Plot a histogram of ROUGE scores",
)
@click.option(
    "--rouge_histplot_fb",
    is_flag=True,
    default=False,
    help="Plot a histogram of ROUGE scores for feedback data",
)
@click.option(
    "--rna_type_distribution",
    is_flag=True,
    default=False,
    help="Plot RNA type distribtion",
)
def main(
    output_path,
    feedback_pair_grid,
    feedback_box_plot,
    feedback_pair_plot,
    rouge_kdeplot,
    rouge_kdeplot_fb,
    rouge_histplot,
    rouge_histplot_fb,
    rna_type_distribution,
):
    output_path = Path(output_path)

    data = pl.read_parquet(
        "with_rouge.parquet"
    )  # .filter(pl.col("selection_method").eq("round-robin")).with_columns(context_size=pl.col("context").str.lengths())
    # .sort("rouge2").filter(pl.col("rouge2").lt(0.71) & pl.col("rougeL").lt(0.8) & pl.col("rouge1").lt(0.8)) #.to_pandas()
    rouge_data = data.melt(
        id_vars="ent_id", value_vars=["rouge1", "rouge2", "rougeL"]
    ).rename({"variable": "rouge-type", "value": "rouge"})

    fb_data = (
        pl.read_parquet("fb_with_rouge.parquet")
        .with_columns(
            pl.col("user_id").apply(lambda x: "Nancy" if x == "anonymous" else x)
        )
        .with_columns(display=pl.col("feedback").gt(2))
    )
    fb_data = anonymise_data(fb_data)
    common_ids = (
        fb_data.groupby("rna_id")
        .agg(pl.col("user_id").unique())
        .filter(pl.col("user_id").list.lengths().gt(2))
        .select("rna_id")
    )

    rouge_data_fb = fb_data.melt(
        id_vars="rna_id", value_vars=["rouge1", "rouge2", "rougeL"]
    ).rename({"variable": "rouge-type", "value": "rouge"})
    rating_data_fb = (
        fb_data.with_columns(
            pl.struct(["user_id", "feedback"])
            .apply(lambda x: {x["user_id"]: x["feedback"]})
            .alias("result")
        )
        .unnest("result")
        .drop("Rater_3")
    )  # .melt(id_vars='rna_id', value_vars=["Rater_0", "Rater_1", "Rater_2"]).rename({"variable": "rater", "value": "rating"}).drop_nulls().unique()
    rating_data_fb = rating_data_fb.join(common_ids, on="rna_id")

    # rating_data_fb = fb_data.groupby("rna_id").agg([pl.col("user_id"), pl.col("feedback")]).melt(id_vars='rna_id', value_vars=["user_id", "feedback"]).rename({"variable": "rater", "value": "rating"})
    # .apply(lambda x: x.with_columns(pl.struct(["user_id", "feedback"]).apply(lambda y: {y['user_id']: y["feedback"]}).alias("result")).unnest("result"))
    print(rating_data_fb)

    if rna_type_distribution:
        rna_type_data = pl.read_parquet("../staging_area/selected_sentences.parquet")

    sns.set_theme(style="darkgrid")

    if feedback_pair_grid:
        print("rater")

        g = sns.PairGrid(
            rating_data_fb.to_pandas(),
            vars=["Rater_0", "Rater_1", "Rater_2"],
            corner=True,
        )
        g.map_diag(sns.barplot)
        g.map_lower(
            scatterFilter, rna_ids=rating_data_fb.get_column("rna_id").to_pandas()
        )
        plt.savefig(output_path / "feedback_pair_grid.png")
        plt.show()

    if feedback_box_plot:
        sns.boxplot(data=fb_data, x="user_id", y="feedback")
        sns.stripplot(
            x="user_id", y="feedback", data=fb_data, size=4, color=".3", linewidth=0
        )
        plt.show()

    if feedback_pair_plot:
        sns.pairplot(rating_data_fb.to_pandas(), vars=["Rater_0", "Rater_1", "Rater_2"])
        plt.show()

    if rouge_kdeplot:
        sns.kdeplot(data, x="rouge1", label="ROUGE-1", common_norm=True)
        sns.kdeplot(data, x="rouge2", label="ROUGE-2", common_norm=True)
        sns.kdeplot(data, x="rougeL", label="ROUGE-L", common_norm=True)
        if rouge_kdeplot_fb:
            sns.kdeplot(fb_data, x="rouge1", label="ROUGE-1", common_norm=True)
            sns.kdeplot(fb_data, x="rouge2", label="ROUGE-2", common_norm=True)
            sns.kdeplot(fb_data, x="rougeL", label="ROUGE-L", common_norm=True)

        plt.xlabel("ROUGE score")
        plt.legend(loc="upper right")
        plt.show()

    if rouge_histplot:
        sns.histplot(
            rouge_data, x="rouge", hue="rouge-type", common_norm=True, element="step"
        )

        if rouge_histplot_fb:
            sns.histplot(fb_data, x="rouge1", label="ROUGE-1", common_norm=True)
            sns.histplot(fb_data, x="rouge2", label="ROUGE-2", common_norm=True)
            sns.histplot(fb_data, x="rougeL", label="ROUGE-L", common_norm=True)

        plt.xlabel("ROUGE score")
        plt.show()

    if rouge_histplot:
        sns.histplot(
            rouge_data, x="rouge", hue="rouge-type", common_norm=True, element="step"
        )

        out_name = "rouge_histplot"
        if rouge_histplot_fb:
            sns.histplot(
                rouge_data_fb,
                x="rouge",
                hue="rouge-type",
                common_norm=True,
                element="step",
            )
            out_name += "_fb"

        plt.xlabel("ROUGE score")
        plt.savefig(output_path / f"{out_name}.png")
        plt.show()

    if rna_type_distribution:
        print("Resolving to simple types")
        so = obonet.read_obo(
            "https://raw.githubusercontent.com/The-Sequence-Ontology/SO-Ontologies/master/Ontology_Files/so.obo"
        )
        lncRNAs = nx.ancestors(so, "SO:0001877")
        lncRNAs.add("SO:0001877")
        miRNAs = nx.ancestors(so, "SO:0000276")
        miRNAs.add("SO:0000276")
        snoRNAs = nx.ancestors(so, "SO:0000275")
        snoRNAs.add("SO:0000275")
        pre_miRNA = nx.ancestors(so, "SO:0001244")
        pre_miRNA.add("SO:0001244")

        print(lncRNAs)
        rna_type_data = rna_type_data.with_columns(
            simple_type=pl.when(pl.col("so_rna_type").is_in(lncRNAs))
            .then("lncRNA")
            .when(pl.col("so_rna_type").is_in(miRNAs))
            .then("miRNA")
            .when(pl.col("so_rna_type").is_in(snoRNAs))
            .then("snoRNA")
            .when(pl.col("so_rna_type").is_in(pre_miRNA))
            .then("pre_miRNA")
            .otherwise("Other")
        )
        print(
            rna_type_data.filter(pl.col("simple_type").eq("Other"))
            .select(["simple_type", "so_rna_type"])
            .unique()
        )

        counts = (
            rna_type_data.groupby("simple_type")
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )
        counts = counts.with_columns(pc=pl.col("count") / rna_type_data.height * 100)
        print(counts)
        sns.barplot(counts.to_pandas(), x="count", y="simple_type")
        plt.ylabel("")
        plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    main()
