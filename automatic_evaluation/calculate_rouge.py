import os

import click
import matplotlib.pyplot as plt
import polars as pl
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def calculate_rouge(row):
    scores = scorer.score(row["context"], row["summary"])

    f1_scores = {k: v.fmeasure for k, v in scores.items()}
    return f1_scores


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--feedback_file", default=None)
def main(input_file, output_file, feedback_file=None):
    data = pl.read_parquet(input_file)

    data = data.with_columns(
        pl.struct(["context", "summary"])
        .apply(
            calculate_rouge,
        )
        .alias("result")
    ).unnest("result")

    data.write_parquet(output_file)

    rouge1 = data.get_column("rouge1").to_numpy()
    rouge2 = data.get_column("rouge2").to_numpy()
    rougeL = data.get_column("rougeL").to_numpy()

    # counts, bins, patches = plt.hist(rouge1, bins=100, label='ROUGE-1')
    # plt.hist(rouge2, bins=bins, label='ROUGE-2')
    # plt.hist(rougeL, bins=bins, label='ROUGE-L')
    # plt.legend(loc='upper right')
    # plt.show()

    if feedback_file is not None:
        fb_data = pl.read_csv(feedback_file)
        fb_data = fb_data.with_columns(
            pl.struct(["context", "summary"])
            .apply(
                calculate_rouge,
            )
            .alias("result")
        ).unnest("result")
        fb_data.write_parquet("fb_with_rouge.parquet")


if __name__ == "__main__":
    main()
