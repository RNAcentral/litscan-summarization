from pathlib import Path

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import obonet
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score


def anonymise_data(data):
    data = data.with_columns(
        pl.col("user_id").apply(lambda x: "Nancy" if x == "anonymous" else x)
    )

    data = data.with_columns(
        pl.when(pl.col("user_id").eq("Nancy"))
        .then("Rater_0")
        .when(pl.col("user_id").eq("blake"))
        .then("Rater_1")
        .when(pl.col("user_id").eq("Andrew"))
        .then("Rater_2")
        .when(pl.col("user_id").eq("Alex"))
        .then("Rater_3")
        .alias("user_id")
    )

    return data


def labelled_hist(x, label, color):
    ax0 = plt.gca()
    ax = ax0.twinx()

    sns.despine(ax=ax, left=True, top=True, right=False)
    ax.yaxis.tick_right()
    ax.set_ylabel("Count")

    ax.hist(x, label=label, color=color)


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
    acc = accuracy_score(x, y)
    print("Kappa | Accuracy")
    print(f"{K:.2f} | {acc:.2f}")

    ax = plt.gca()
    ax.plot(
        x + np.random.normal(0, 0.1, interimDf.height),
        y + np.random.normal(0, 0.1, interimDf.height),
        ".",
        **kwargs,
    )
    ax.set_xlim([0, 6])
    ax.set_ylim([0, 6])
    ax.text(1, 6, "$\kappa$ = " + str(round(K, 2)))
    ax.text(1, 5.5, "acc = " + str(round(acc, 2)))


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
    "--bert_histplot",
    is_flag=True,
    default=False,
    help="Plot a histogram of BERT scores",
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
@click.option(
    "--feedback_correlation",
    is_flag=True,
    default=False,
    help="Plot correlation between feedback ratings",
)
@click.option(
    "--feedback_correlation_bert",
    is_flag=True,
    default=False,
    help="Plot correlation between feedback ratings",
)
@click.option(
    "--bert_rouge_correlation",
    is_flag=True,
    default=False,
    help="Plot correlation between feedback ratings",
)
@click.option(
    "--rna_type_feedback",
    is_flag=True,
    default=False,
    help="Plot the per-rater feedback, and average feedback grouped by RNA type",
)
@click.option(
    "--feedback_histograms",
    is_flag=True,
    default=False,
    help="Plot a pair plot of feedback data",
)
@click.option("--gpt4", is_flag=True, default=False, help="Use GPT4 summary ratings")
@click.option(
    "--checkboxes",
    is_flag=True,
    default=False,
    help="Run aggregations to get checkbox stats",
)
def main(
    output_path,
    feedback_pair_grid,
    feedback_box_plot,
    feedback_pair_plot,
    rouge_kdeplot,
    rouge_kdeplot_fb,
    rouge_histplot,
    bert_histplot,
    rouge_histplot_fb,
    rna_type_distribution,
    feedback_correlation,
    feedback_correlation_bert,
    bert_rouge_correlation,
    rna_type_feedback,
    feedback_histograms,
    gpt4,
    checkboxes,
):
    # pl.Config.set_tbl_rows(1000)
    pl.Config.set_tbl_cols(1000)
    output_path = Path(output_path)

    basedir = "/Users/agreen/code/litscan-summarization/automatic_evaluation"

    # data = pl.read_parquet(
    #     f"{basedir}/with_rouge.parquet"
    # )  # .filter(pl.col("selection_method").eq("round-robin")).with_columns(context_size=pl.col("context").str.lengths())
    # # .sort("rouge2").filter(pl.col("rouge2").lt(0.71) & pl.col("rougeL").lt(0.8) & pl.col("rouge1").lt(0.8)) #.to_pandas()
    # rouge_data = data.melt(
    #     id_vars="ent_id", value_vars=["rouge1", "rouge2", "rougeL"]
    # ).rename({"variable": "rouge-type", "value": "rouge"})

    bert_data = pl.read_parquet(f"{basedir}/with_bert.parquet").with_columns(
        dummy=pl.lit("f1")
    )
    bert_data_fb = (
        pl.read_parquet(f"{basedir}/fb_with_bert.parquet")
        .with_columns(
            pl.col("user_id").apply(lambda x: "Nancy" if x == "anonymous" else x)
        )
        .with_columns(display=pl.col("feedback").gt(2))
        .with_columns(dummy=pl.lit("f1"))
        .rename({"ent_id": "rna_id"})
        .select(["rna_id", "f1"])
        .unique("rna_id")
    )

    rouge_data_fb = (
        pl.read_parquet(f"{basedir}/fb_with_rouge.parquet")
        .with_columns(
            pl.col("user_id").apply(lambda x: "Nancy" if x == "anonymous" else x)
        )
        .with_columns(display=pl.col("feedback").gt(2))
        .with_columns(dummy=pl.lit("rouge"))
        # .rename({"ent_id": "rna_id"})
        .select(["rna_id", "rouge1", "rouge2", "rougeL"])
        .unique("rna_id")
    )

    if gpt4:
        fb_data = (
            pl.read_parquet("all_feedback_GPT4.pq")
            .with_columns(display=pl.col("feedback").gt(2))
            .with_columns(dummy=pl.lit("1"))
        )
        fb_data = fb_data.filter(pl.col("user_id") != "None")
        fb_data = fb_data.unique(subset=["rna_id", "user_id"]).sort(by="summary_id")
    else:
        fb_data = (
            pl.read_parquet(f"{basedir}/all_feedback_200.parquet")
            .with_columns(
                pl.col("user_id").apply(lambda x: "Nancy" if x == "anonymous" else x)
            )
            .with_columns(display=pl.col("feedback").gt(2))
            .with_columns(dummy=pl.lit("1"))
        )

    fb_data = anonymise_data(fb_data)
    average_rating = fb_data.groupby("summary_id").agg(
        pl.col("feedback").mean().alias("average")
    )

    second_batch = (
        fb_data.join(bert_data_fb, on="rna_id")
        .join(rouge_data_fb, on="rna_id")
        .filter(pl.col("summary_id").gt(300))
        .with_columns(dummy=pl.lit("f1"))
        .select(
            [
                "rna_id",
                "feedback",
                "user_id",
                "f1",
                "rouge1",
                "rouge2",
                "rougeL",
                "dummy",
            ]
        )
        .unique(["rna_id", "user_id"])
        .drop_nulls("user_id")
    )

    # bert_data_fb = anonymise_data(bert_data_fb)

    common_ids = (
        fb_data.groupby("rna_id")
        .agg(pl.col("user_id").unique())
        .filter(pl.col("user_id").list.lengths().gt(2))
        .select("rna_id")
    )

    rouge_data_fb = (
        second_batch.melt(id_vars="rna_id", value_vars=["rouge1", "rouge2", "rougeL"])
        .rename({"variable": "rouge-type", "value": "rouge"})
        .join(second_batch, on="rna_id")
        .select(["rna_id", "rouge-type", "rouge", "feedback", "user_id"])
        .filter(pl.col("user_id").is_in(["Rater_0", "Rater_1", "Rater_2"]))
        .unique(["rna_id", "rouge-type", "rouge", "user_id"])
    )

    rating_data_fb = (
        fb_data.filter(pl.col("summary_id").gt(300))
        .with_columns(
            pl.struct(["user_id", "feedback"])
            .apply(lambda x: {x["user_id"]: x["feedback"]})
            .alias("result")
        )
        .unnest("result")
        # .drop("Rater_3")
    )  # .melt(id_vars='rna_id', value_vars=["Rater_0", "Rater_1", "Rater_2"]).rename({"variable": "rater", "value": "rating"}).drop_nulls().unique()
    rating_data_fb = rating_data_fb.join(common_ids, on="rna_id")

    # rating_data_fb = fb_data.groupby("rna_id").agg([pl.col("user_id"), pl.col("feedback")]).melt(id_vars='rna_id', value_vars=["user_id", "feedback"]).rename({"variable": "rater", "value": "rating"})
    # .apply(lambda x: x.with_columns(pl.struct(["user_id", "feedback"]).apply(lambda y: {y['user_id']: y["feedback"]}).alias("result")).unnest("result"))

    if rna_type_distribution:
        rna_type_data = pl.read_parquet("../staging_area/selected_sentences.parquet")

    sns.set_theme(style="whitegrid")

    if feedback_pair_grid:
        print("rater")

        g = sns.PairGrid(
            rating_data_fb.to_pandas(),
            vars=["Rater_0", "Rater_1", "Rater_2"],
            corner=True,
        )
        g.map_diag(labelled_hist)
        g.map_lower(
            scatterFilter, rna_ids=rating_data_fb.get_column("rna_id").to_pandas()
        )

        for ax in g.axes.flat:
            if ax:
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.set_yticks([1, 2, 3, 4, 5])
        plt.tight_layout()
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

    if bert_histplot:
        sns.histplot(bert_data, x="f1", common_norm=True, element="step")

        out_name = "bert_histplot"
        # if rouge_histplot_fb:
        #     sns.histplot(
        #         rouge_data_fb,
        #         x="rouge",
        #         hue="rouge-type",
        #         common_norm=True,
        #         element="step",
        #     )
        #     out_name += "_fb"

        plt.xlabel("BERT score")
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
        print(counts.sum())
        ax = sns.barplot(counts.to_pandas(), x="count", y="simple_type")
        plt.ylabel("")

        labels_large = [
            f"{t} ({(t/4674)*100:.2f}%)" if t > 1000 else ""
            for t in counts.get_column("count").to_numpy()
        ]
        labels_small = [
            f"{t} ({(t/4674)*100:.2f}%)" if t < 1000 else ""
            for t in counts.get_column("count").to_numpy()
        ]
        ax.bar_label(
            ax.containers[0], labels_large, padding=-90, color="white"
        )  # , fmt="%.0f")
        ax.bar_label(ax.containers[0], labels_small, padding=5)
        plt.savefig(output_path / "RNA_type_distribution.png", bbox_inches="tight")
        # plt.tight_layout()
        plt.show()

    if feedback_correlation:
        g = sns.FacetGrid(
            rouge_data_fb.to_pandas(),
            col="rouge-type",
            row="user_id",
            row_order=["Rater_0", "Rater_1", "Rater_2"],
            col_order=["rouge1", "rouge2", "rougeL"],
            margin_titles=True,
        )
        g.map(plt.scatter, "rouge", "feedback")
        g.set_axis_labels("ROUGE score", "Feedback rating")
        g.set(ylim=(0.5, 6))
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        for (row_val, col_val), ax in g.axes_dict.items():
            print(row_val, col_val)
            corr_data = (
                rouge_data_fb.filter(pl.col("rouge-type").eq(col_val))
                .filter(pl.col("user_id").eq(row_val))
                .unique(["rna_id", "user_id"])
                .select(["rouge", "feedback"])
            )
            fb = corr_data.get_column("feedback").to_numpy()
            rouge = corr_data.get_column("rouge").to_numpy()
            corr_obj = spearmanr(rouge, fb)
            print(corr_obj.correlation)
            ax.text(
                0.1,
                5.5,
                r"$\rho$ = {}, p = {}".format(
                    round(corr_obj.correlation, 2), round(corr_obj.pvalue, 2)
                ),
            )

        g.tight_layout()
        plt.savefig(output_path / "rouge_feedback_correlation.png")
        plt.show()

    if feedback_correlation_bert:
        g = sns.FacetGrid(
            second_batch.to_pandas(),
            col="user_id",
            row="dummy",
            col_order=["Rater_0", "Rater_1", "Rater_2"],
            row_order=["f1"],
            margin_titles=True,
        )
        g.map(plt.scatter, "f1", "feedback")
        g.set_axis_labels("BERT score", "Feedback rating")
        g.set(ylim=(0.5, 6))
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        for (row_val, col_val), ax in g.axes_dict.items():
            print(row_val, col_val)
            corr_data = (
                second_batch.filter(pl.col("user_id").eq(col_val))
                .unique(["rna_id", "user_id"])
                .select(["f1", "feedback"])
            )
            fb = corr_data.get_column("feedback").to_numpy()
            f1 = corr_data.get_column("f1").to_numpy()
            corr_obj = spearmanr(f1, fb)
            print(corr_obj.correlation)
            ax.text(
                0.86,
                5.5,
                r"$\rho$ = {}, p = {}".format(
                    round(corr_obj.correlation, 2), round(corr_obj.pvalue, 2)
                ),
            )

        g.tight_layout()
        plt.savefig(output_path / "BERTscore_feedback_correlation.png")
        plt.show()

    if bert_rouge_correlation:
        print(rouge_data)
        combodata = bert_data.select(["ent_id", "f1", "dummy"]).join(
            rouge_data, left_on="ent_id", right_on="ent_id"
        )
        print(
            combodata.filter(pl.col("rouge-type").eq("rouge1")).select(["f1", "rouge"])
        )
        g = sns.FacetGrid(
            combodata.to_pandas(),
            col="rouge-type",
            row="dummy",
            col_order=["rouge1", "rouge2", "rougeL"],
            row_order=["f1"],
            margin_titles=True,
        )
        g.map(plt.scatter, "rouge", "f1")
        g.set_axis_labels("ROUGE score", "BERT score")
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        for (row_val, col_val), ax in g.axes_dict.items():
            print(row_val, col_val)
            corr_data = (
                combodata.filter(pl.col("rouge-type").eq(col_val))
                .unique(["ent_id"])
                .select(["f1", "rouge"])
            )
            fb = corr_data.get_column("rouge").to_numpy()
            f1 = corr_data.get_column("f1").to_numpy()
            corr_obj = spearmanr(f1, fb)
            print(corr_obj.correlation)
            ax.text(
                0.055,
                0.98,
                r"$\rho$ = {}, p = {}".format(
                    round(corr_obj.correlation, 2), round(corr_obj.pvalue, 2)
                ),
            )
        plt.savefig(output_path / "BERTscore_ROUGE_correlation.png")

        plt.show()

    if feedback_histograms:
        # g = sns.FacetGrid(
        #     rating_data_compare.to_pandas(),
        #     col="user_id",
        #     row="batch",
        #     col_order=["Rater_0", "Rater_1", "Rater_2"],
        #     # row_order=["feedback"],
        #     margin_titles=False,
        # )
        # g.map_dataframe(sns.histplot,  x="feedback", bins=[0,1,2,3,4,5,6],discrete=True)
        print(fb_data.columns)
        g = sns.FacetGrid(
            fb_data.to_pandas(),
            col="user_id",
            row="dummy",
            col_order=["Rater_0", "Rater_1", "Rater_2"],
            # row_order=["feedback"],
            margin_titles=False,
        )
        g.map_dataframe(
            sns.histplot, x="feedback", bins=[0, 1, 2, 3, 4, 5, 6], discrete=True
        )
        g.set_titles(template="{col_name}")
        for ax in g.axes.flat:
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel("Feedback")
        plt.savefig(output_path / "feedback_histograms.png")
        # g.facet_axis(1, 0, modify_state=True)

        plt.figure()
        from matplotlib.ticker import MaxNLocator

        ax = sns.histplot(
            average_rating.to_pandas(),
            x="average",
            bins=[0, 1, 2, 3, 4, 5, 6],
            discrete=True,
        )
        ticks = [np.ceil(y) for y in ax.get_yticks()]
        ax.set_yticks(ticks)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim((0, 6))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Average Feedback")
        plt.savefig(output_path / "feedback_histograms_average.png")
        plt.show()

        modes = [
            "hallucinated",
            "not supported",
            "contradiction",
            "format",
            "causality",
        ]

        ## Everyone has a slightly different way of recording these things...
        failures = fb_data.filter(pl.col("feedback").lt(3))
        a_failures = failures.filter(pl.col("user_id").eq("Rater_2"))
        a_failures = a_failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*hallucin.*ref.*")
            .alias("reference formatting")
        )
        a_failures = a_failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*hallucin.*")
            .__and__(pl.col("reference formatting").is_not())
            .alias("hallucinated")
        )
        a_failures = a_failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*causality.*")
            .alias("causality")
        )
        a_failures = a_failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*contra.*")
            .alias("contradiction")
        )
        a_failures = a_failures.with_columns(
            pl.col("contradiction")
            .__or__(pl.col("causality"))
            .__or__(pl.col("inaccurate_text"))
            .alias("inaccurate text")
        )
        a_failures = a_failures.drop(["contradiction", "causality"])

        b_failures = failures.filter(pl.col("user_id").eq("Rater_1"))
        b_failures = b_failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*hallucin.*ref.*")
            .alias("reference formatting")
        )
        b_failures = b_failures.with_columns(
            pl.col("contains_hallucinations").alias("hallucinated")
        )
        b_failures = b_failures.with_columns(
            pl.col("inaccurate_text").alias("inaccurate text")
        )

        print(b_failures.get_column("feedback_id").to_list())

        c_failures = failures.filter(pl.col("user_id").eq("Rater_0"))
        c_failures = c_failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*not supported.*")
            .alias("inaccurate text")
        )
        c_failures = c_failures.with_columns(
            pl.col("inaccurate_text")
            .__or__(pl.col("inaccurate text"))
            .alias("inaccurate text")
        )
        c_failures = c_failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*format.*")
            .alias("Reference formatting")
        )

        print(a_failures)
        print(b_failures)
        print(c_failures)
        exit()

        failures = failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*hallucin.*")
            .alias("hallucinated")
        )
        failures = failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*not supported.*")
            .alias("not supported")
        )
        failures = failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*contra.*")
            .alias("contradiction")
        )
        failures = failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*format.*")
            .alias("reference formatting")
        )
        failures = failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*causality.*")
            .alias("causality")
        )
        failures = failures.with_columns(
            pl.col("free_feedback")
            .str.to_lowercase()
            .str.contains(".*irrelevant.*")
            .alias("irrelevant detail")
        )
        failures = failures.with_columns(
            pl.col("contradiction")
            .__or__(
                pl.col("not supported")
                .__or__(pl.col("causality"))
                .__or__(pl.col("inaccurate_text"))
                .__or__(pl.col("hallucinated"))
            )
            .alias("inaccurate text")
        )
        failures = failures.with_columns(
            pl.col("hallucinated").__or__(pl.col("contains_hallucinations"))
        )
        failures = failures.with_columns(
            pl.col("irrelevant detail").__or__(pl.col("over_specific"))
        )

        print(
            failures.select(
                pl.col("feedback_id"),
                pl.col("hallucinated"),
                pl.col("not supported"),
                pl.col("reference formatting"),
                pl.col("irrelevant detail"),
            )
        )

        print(failures.groupby("user_id").agg(pl.count(), pl.col("feedback_id")))
        for fb in (
            failures.filter(pl.col("user_id").eq("Rater_1"))
            .get_column("free_feedback")
            .to_list()
        ):
            print(fb)

        fail_type_count = failures.groupby("user_id").agg(
            pl.col("hallucinated"),
            pl.col("reference formatting"),
            pl.col("irrelevant detail"),
        )
        fail_type_count = fail_type_count.with_columns(
            pl.col("hallucinated").list.count_match(True),
            # pl.col("inaccurate_text").list.count_match(True),
            pl.col("reference formatting").list.count_match(True),
            pl.col("irrelevant detail").list.count_match(True),
        )
        print(fail_type_count)
        fail_type_count_bar = fail_type_count.melt(id_vars="user_id")
        print(fail_type_count)
        plt.figure()
        sns.barplot(
            fail_type_count_bar.to_pandas(), x="variable", y="value", hue="user_id"
        )
        plt.xlabel("Failure Mode")
        plt.ylabel("Count")

        # plt.figure()
        # sns.heatmap(fail_type_count.select(["hallucinated", "inaccurate_text", "formatting", "irrelevant_detail"]).to_pandas(), annot=True)
        plt.show()

    if rna_type_feedback:
        ## read and prepare feedback data
        # fb_data = (
        # pl.read_parquet("all_feedback_200.parquet")
        # .with_columns(
        #     pl.col("user_id").apply(lambda x: "Nancy" if x == "anonymous" else x)
        # )
        # .with_columns(display=pl.col("feedback").gt(2))
        # )
        # fb_data = anonymise_data(fb_data)

        second_batch = fb_data.filter(pl.col("summary_id").gt(300))

        ## Read and prep RNA type data
        rna_type_data = pl.read_parquet("../staging_area/selected_sentences.parquet")

        print("Resolving to simple types")
        # "https://raw.githubusercontent.com/The-Sequence-Ontology/SO-Ontologies/master/Ontology_Files/so.obo"
        so = obonet.read_obo(
            "https://raw.githubusercontent.com/The-Sequence-Ontology/SO-Ontologies/07b30b453295147efa9f9f8c017907cd147fcaa9/Ontology_Files/so.obo"
        )
        lncRNAs = nx.ancestors(so, "SO:0001877")
        lncRNAs.add("SO:0001877")
        miRNAs = nx.ancestors(so, "SO:0000276")
        miRNAs.add("SO:0000276")
        snoRNAs = nx.ancestors(so, "SO:0000275")
        snoRNAs.add("SO:0000275")
        pre_miRNA = nx.ancestors(so, "SO:0001244")
        pre_miRNA.add("SO:0001244")

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
        ).with_columns(pl.col("primary_id").str.to_lowercase())
        type_rating_data = rna_type_data.join(
            rating_data_fb, left_on="primary_id", right_on="rna_id"
        )
        print(type_rating_data)
        average_rating = type_rating_data.groupby("simple_type").agg(
            [pl.col("feedback").count().alias("count"), pl.col("primary_id")]
        )
        average_rating = average_rating.explode("primary_id")
        print(average_rating.unique("simple_type"))
        print(rating_data_fb.unique("summary_id"))

        type_rating_data = type_rating_data.join(average_rating, on="primary_id").sort(
            by="count", descending=True
        )
        type_rating_data = type_rating_data.unique(
            subset=["urs_taxid", "Rater_0", "Rater_1", "Rater_2"]
        )
        print(
            type_rating_data.select(
                ["urs_taxid", "Rater_0", "Rater_1", "Rater_2"]
            ).sort(by="urs_taxid")
        )  # .unique("simple_type", maintain_order=True))

        g = sns.FacetGrid(
            type_rating_data.to_pandas(),
            col="user_id",
            row="simple_type",
            col_order=["Rater_0", "Rater_1", "Rater_2"],
            hue="simple_type",
            hue_order=["lncRNA", "miRNA", "pre_miRNA", "snoRNA", "Other"],
            # row_order=["count"],
            margin_titles=False,
        )
        g.map_dataframe(
            sns.histplot, x="feedback", bins=[0, 1, 2, 3, 4, 5, 6], discrete=True
        )
        g.set_titles(template="{col_name}")
        for ax in g.axes.flat:
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel("Feedback")

        ## manually set the right hand axis label for rna type
        type_names = type_rating_data.unique(
            "simple_type", maintain_order=True
        ).get_column("simple_type")
        for i, type_name in enumerate(type_names):
            g.axes[i][0].set_ylabel(type_name)

        for row in range(1, len(type_names)):
            for col in range(3):
                g.axes[row][col].set_title("")
        # g.axes[0][0].set_ylabel("lncRNA")
        # g.axes[1][0].set_ylabel("pre_miRNA")
        # g.axes[2][0].set_ylabel("snoRNA")
        # g.axes[3][0].set_ylabel("miRNA")
        # g.axes[4][0].set_ylabel("Other")

        # plt.savefig(output_path / "feedback_histograms.png")
        # g.facet_axis(1, 0, modify_state=True)

        # plt.figure()
        # from matplotlib.ticker import MaxNLocator
        # ax = sns.histplot(average_rating.to_pandas(), x="average", bins=[0,1,2,3,4,5,6],discrete=True)
        # # ticks = [np.ceil(y) for y in ax.get_yticks()]
        # # ax.set_yticklabels(ticks)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.set_xlabel("Average Feedback")
        plt.savefig(output_path / "feedback_histograms_type.png")
        plt.show()

        exit()

        print(fb_data)
        print(rna_type_data)
        print(fb_data.columns)
        print(rna_type_data.columns)
        exit()
        # rna_type_fb_data = rna_type_data.join(fb_data, left_on=)

        averages = (
            rna_type_fb_data.groupby("simple_type")
            .agg(pl.mean("feedback").alias("count"))
            .sort("count", descending=True)
        )
        counts = counts.with_columns(pc=pl.col("count") / rna_type_data.height * 100)
        print(counts.sum())
        ax = sns.barplot(counts.to_pandas(), x="count", y="simple_type")
        plt.ylabel("")

        labels_large = [
            f"{t} ({(t/4674)*100:.2f}%)" if t > 1000 else ""
            for t in counts.get_column("count").to_numpy()
        ]
        labels_small = [
            f"{t} ({(t/4674)*100:.2f}%)" if t < 1000 else ""
            for t in counts.get_column("count").to_numpy()
        ]
        ax.bar_label(
            ax.containers[0], labels_large, padding=-90, color="white"
        )  # , fmt="%.0f")
        ax.bar_label(ax.containers[0], labels_small, padding=5)
        # plt.savefig(output_path / "RNA_type_feedback_distribution.png", bbox_inches="tight")
        # plt.tight_layout()
        plt.show()

    if checkboxes:
        print(fb_data.columns)
        print(
            f"False positives:{fb_data.filter(pl.col('false_positive') == 't').height}"
        )
        print(
            f"Hallucinations:{fb_data.filter(pl.col('contains_hallucinations') == 't').height}"
        )
        print(
            f"Inaccurate Text:{fb_data.filter(pl.col('inaccurate_text') == 't').height}"
        )
        print(f"Contradictions:{fb_data.filter(pl.col('contradictory') == 't').height}")
        print(f"Over Specific:{fb_data.filter(pl.col('over_specific') == 't').height}")
        print(f"Bad Length:{fb_data.filter(pl.col('bad_length') == 't').height}")
        print(f"Mentions AI:{fb_data.filter(pl.col('mentions_ai') == 't').height}")
        print(f"Short Context:{fb_data.filter(pl.col('short_context') == 't').height}")

        for summary_id in fb_data.get_column("summary_id").unique().to_list():
            feedbacks = (
                fb_data.filter(pl.col("summary_id") == summary_id)
                .get_column("free_feedback")
                .to_list()
            )
            checks = fb_data.filter(pl.col("summary_id") == summary_id).select(
                [
                    "user_id",
                    "feedback",
                    "false_positive",
                    "contains_hallucinations",
                    "inaccurate_text",
                    "contradictory",
                    "over_specific",
                    "bad_length",
                    "mentions_ai",
                    "short_context",
                ]
            )
            print(checks)
            print("\n\n".join(feedbacks))

            n = input("input something to continue")


if __name__ == "__main__":
    main()
