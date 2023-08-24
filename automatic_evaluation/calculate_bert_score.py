import os

import click
import polars as pl
from bert_score import score

model = "allenai/longformer-base-4096"


def calculate_bert_score(row, device, batch_size):
    row.get_column("context").to_numpy(), row.get_column("summary").to_numpy()
    contexts = [c.strip() for c in contexts]
    summaries = [s.strip() for s in summaries]
    P, R, F1 = score(
        contexts,
        summaries,
        lang="en",
        verbose=True,
        model_type=model,
        device=device,
        batch_size=batch_size,
    )
    return {"precision": P, "recall": R, "f1": F1}


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--device", default="cpu")
@click.option("--batch_size", default=16)
def main(input_file, output_file, device, batch_size):
    data = pl.read_parquet(input_file)
    print(data)
    data = data.with_columns(
        result=pl.struct(["context", "summary"]).apply(
            lambda x: calculate_bert_score(x, device, batch_size)
        )
    ).unnest("result")

    data.write_parquet(output_file)


if __name__ == "__main__":
    main()
