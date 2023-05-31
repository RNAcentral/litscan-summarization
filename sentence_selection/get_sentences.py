import os
import typing as ty
from pathlib import Path

import polars as pl

from sentence_selection.aliases import resolve_aliases
from sentence_selection.pull_sentences import pull_data_from_db
from sentence_selection.sentence_selector import iterative_sentence_selector
from sentence_selection.utils import get_token_length

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sentence_transformers import SentenceTransformer


def sample_sentences(
    sentences: pl.DataFrame, model: SentenceTransformer, limit: ty.Optional[int] = 3072
):

    df = sentences.with_columns(
        pl.struct(["sentence", "primary_id", "pmcid"])
        .apply(lambda x: iterative_sentence_selector(x, model, limit))
        .alias("result")
    ).unnest("result")
    return df


def tokenize_and_count(sentence_df: pl.DataFrame, limit: ty.Optional[int] = 3072):
    df = sentence_df.with_columns(
        [pl.col("sentence").apply(get_token_length).alias("num_tokens")]
    )
    df = df.with_columns(
        [
            pl.col("num_tokens").arr.sum().alias("total"),
            pl.col("pmcid").arr.lengths().alias("num_articles"),
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
    if Path(cache).exists():
        sentence_df = pl.read_json(cache)
    else:
        sentence_df = pull_data_from_db(conn_str, query)
        sentence_df = tokenize_and_count(sentence_df)
        sentence_df.write_json(cache)
    sentence_df = resolve_aliases(sentence_df)

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    sample_df = sample_sentences(sentence_df, model, limit)

    return sample_df.select(["primary_id", "selected_pmcids", "selected_sentences"])
