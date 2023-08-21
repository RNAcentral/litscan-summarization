import os

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from bert_score import score
from scipy.stats import spearmanr

model = "allenai/longformer-base-4096"


def calculate_bert_score(contexts, summaries):
    contexts = [c.strip() for c in contexts]
    summaries = [s.strip() for s in summaries]
    P, R, F1 = score(
        contexts,
        summaries,
        lang="en",
        verbose=True,
        model_type=model,
        device="mps",
        batch_size=1,
    )
    return {"precision": P, "recall": R, "f1": F1}


@click.command()
@click.argument("input_file")
@click.argument("output_file")
def main(input_file, output_file):
    data = pl.read_parquet(input_file)
    print(data)
    bert_scores = calculate_bert_score(
        data.get_column("context").to_numpy(), data.get_column("summary").to_numpy()
    )

    counts, bins, patches = plt.hist(bert_scores["f1"], bins=100, label="BERTScore")
    plt.show()


if __name__ == "__main__":
    main()
