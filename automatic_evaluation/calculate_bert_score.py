import os

import click
import polars as pl
from bert_score import score

model = "allenai/longformer-base-4096"


def calculate_bert_score(row, device, batch_size):
    contexts = row.get_column("context").to_numpy()
    summaries = row.get_column("summary").to_numpy()
    print(contexts)
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
    return {
        "ent_id": row.get_column("ent_id"),
        "precision": pl.Series(P.numpy()),
        "recall": pl.Series(R.numpy()),
        "f1": pl.Series(F1.numpy()),
    }


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--device", default="cpu")
@click.option("--batch_size", default=16)
def main(input_file, output_file, device, batch_size):
    data = pl.read_parquet(input_file)
    print(data)
    result = pl.DataFrame(
        calculate_bert_score(
            data.select(["ent_id", "context", "summary"]), device, batch_size
        )
    )
    data = data.join(result, on="ent_id")
    print(data)


if __name__ == "__main__":
    main()
