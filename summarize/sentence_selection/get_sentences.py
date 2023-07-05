import os
import typing as ty
from pathlib import Path

import polars as pl
from sentence_selection.aliases import resolve_aliases
from sentence_selection.pull_sentences import pull_data_from_db
from sentence_selection.sentence_selector import iterative_sentence_selector
from sentence_selection.utils import get_token_length
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sentence_transformers import SentenceTransformer


def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo


def sample_sentences(
    sentences: pl.DataFrame, model: SentenceTransformer, limit: ty.Optional[int] = 3072
):
    pbar = tqdm(total=len(sentences), desc="Selecting Sentences...", colour="green")
    df = sentences.with_columns(
        pl.struct(["sentence", "primary_id", "pmcid"])
        .apply(w_pbar(pbar, lambda x: iterative_sentence_selector(x, model, limit)))
        .alias("result")
    ).unnest("result")
    return df


def tokenize_and_count(sentence_df: pl.DataFrame, limit: ty.Optional[int] = 3072):
    df = sentence_df.with_columns(
        [pl.col("sentence").apply(get_token_length).alias("num_tokens")]
    )
    df = df.with_columns(
        [
            pl.col("num_tokens").list.sum().alias("total"),
            pl.col("pmcid").list.lengths().alias("num_articles"),
        ]
    ).sort("total", descending=True)
    print(
        f"Number of entities with fewer than {limit} total tokens: {df.filter(pl.col('total').lt(limit)).height} before selection"
    )
    return df


def for_summary(
    conn_str: str,
    query: str,
    cache: Path,
    device: ty.Optional[str] = "cpu:0",
    limit: ty.Optional[int] = 3072,
) -> pl.DataFrame:
    if cache is not None and Path(cache).exists():
        sentence_df = pl.read_json(cache)
    else:
        sentence_df = pull_data_from_db(conn_str, query)
        if len(sentence_df) == 0:
            ## No sentences to summarize
            return None

        sentence_df = tokenize_and_count(sentence_df)
        if cache is not None:
            sentence_df.write_json(cache)
    sentence_df = resolve_aliases(sentence_df)

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # sentence_df = sentence_df.filter(pl.col("primary_id").is_in(["FAM197Y2P", "LOC100288254", "FLJ13224", "LINC00910", "PSMA3-AS1", "SLC25A25-AS1"]))
    # ## WIP: Dealing with huge numbers of sentences is hard
    # ## First deal with those below context limit
    under_limit = sentence_df.filter(pl.col("total").lt(limit))
    ## Get the remainder with an anti join
    remainder = sentence_df.join(under_limit, on="primary_id", how="anti")
    if len(under_limit) > 0:
        tiny_df = sample_sentences(under_limit, model, limit)
        tiny_df = tiny_df.with_columns(multiple=pl.lit(0).cast(pl.Int64))
        print(f"Dataframe below context limit is size {len(tiny_df)}")

    # ## Count what multiple each is of the limit
    remainder = remainder.with_columns(
        multiple=(pl.col("total") / limit).floor().cast(pl.Int64)
    )
    # ## Partition by multiple of the limit - should give manageable chunks
    partitions = sorted(
        remainder.partition_by("multiple"), key=lambda x: len(x), reverse=True
    )
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
    return sample_df.select(
        ["primary_id", "selected_pmcids", "selected_sentences", "method", "multiple"]
    )
