from pathlib import Path

import click
import polars as pl
from sentence_selection.aliases import resolve_aliases
from sentence_selection.get_sentences import sample_sentences, tokenize_and_count
from sentence_transformers import SentenceTransformer


@click.command()
@click.option(
    "--raw_sentences",
    type=click.Path(),
    help="Path to raw sentences, should be a parquet file",
)
@click.option(
    "--output",
    default="selected_sentences.parquet",
    type=click.Path(),
    help="Path to output file, should be a parquet file",
)
@click.option("--device", default="cpu:0", help="Device to use for sentence selection")
@click.option("--limit", default=3072, help="Maximum number of tokens to select")
def main(raw_sentences, output, device, limit):
    """
    Take the sentences extracted from the database and run topic modelling over them to select sentences to fill the context.
    """

    raw_sentences = pl.read_parquet(raw_sentences)

    ## Add the token counts for all sentences
    raw_sentences = tokenize_and_count(raw_sentences)

    ## Regex replace aliases with primary id in all sentences
    raw_sentences = resolve_aliases(raw_sentences)

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    ## First deal with those below context limit
    under_limit = raw_sentences.filter(pl.col("total").lt(limit))

    ## Calculate the remainder
    remainder = raw_sentences.join(under_limit, on="primary_id", how="anti")

    ## Run 'sampling' on the under_limit sentences
    if len(under_limit) > 0:
        tiny_df = sample_sentences(under_limit, model, limit)
        tiny_df = tiny_df.with_columns(multiple=pl.lit(0).cast(pl.Int64))
        print(f"Dataframe below context limit is size {len(tiny_df)}")

    # ## Count what multiple each is of the limit
    remainder = remainder.with_columns(
        multiple=(pl.col("total") / limit).floor().cast(pl.Int64)
    )
    ## Partition by multiple of the limit - should give manageable chunks
    partitions = sorted(
        remainder.partition_by("multiple"), key=lambda x: len(x), reverse=True
    )

    ## Run selection on each partition in turn
    N_part = len(partitions)
    for num, partition in enumerate(partitions):
        print(
            f"Dataframe with {partition.get_column('multiple').unique().to_numpy()[0]}x context limit is size {len(partition)}"
        )
        intermediate = sample_sentences(partition, model, limit)
        intermediate.write_json(f"intermediate_{num}.json")
        del intermediate  ## I don't think this is actually that big, but just in case

    ## Reassemble
    if len(under_limit) > 0:
        sample_df = tiny_df
        start = 0
    else:
        sample_df = pl.read_json(f"intermediate_0.json")
        start = 1
        Path(f"intermediate_0.json").unlink()

    for num in range(start, N_part):
        sample_df = sample_df.vstack(
            pl.read_json(f"intermediate_{num}.json").select(sample_df.columns)
        )
        Path(f"intermediate_{num}.json").unlink()

    print(sample_df)
    sample_df.write_parquet(output)


if __name__ == "__main__":
    main()
