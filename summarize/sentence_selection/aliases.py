"""
Here, we take the datafram with per-ID sentences and then groupby on the primary id
"""

import polars as pl


def find_replace_ids(row):
    """
    Find and replace IDs in a row
    """
    find_re = f"(?i){'|'.join(row['job_id'])}"
    replace_re = f"{row['primary_id']}"
    sentences = pl.Series(row["sentence"])
    sentences = sentences.str.replace_all(find_re, replace_re, literal=False)

    return sentences


def resolve_aliases(df):
    df = df.with_columns(
        sentence=pl.struct(["sentence", "primary_id", "job_id"]).apply(find_replace_ids)
    )
    return df
