import os

import numpy as np
import polars as pl
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

QUERY = """
select result_id, lsa.pmcid as pmcid,
				lsr.job_id as job_id,
				sentence
from embassy_rw.litscan_body_sentence lsb
join embassy_rw.litscan_result lsr on lsr.id = lsb.result_id
join embassy_rw.litscan_database lsdb on lsdb.job_id = lsr.job_id
join embassy_rw.litscan_article lsa on lsa.pmcid = lsr.pmcid

where location = 'intro'
-- and name in ('pombase', 'hgnc', 'wormbase', 'mirbase')
and retracted = false
and lsr.job_id not in ('12s', '12s rrna', '12 s rrna',
                       '13a', '16s', '16s rna',
                       '16srrna', '16s rrna',
                       '2a-1', '2b-2', '45s pre-rrna', '7sk',
                       '7sk rna', '7sk snrna', '7slrna',
                       '7sl rna', 'trna', 'snrna', 'mpa', 'msa', 'rns', 'tran')
"""


def get_token_length(sentences):
    return [len(enc.encode(s)) for s in sentences]


def get_sentences():
    conn_str = os.getenv("PGDATABASE")
    df = pl.read_database(QUERY, conn_str)

    filtered = (
        df.groupby(["job_id"])
        .agg(pl.col("*"))
        .filter(pl.col("result_id").arr.lengths() > 1)
    )

    lengths = (
        filtered.select(pl.col("result_id").arr.unique().arr.lengths())
        .to_numpy()
        .flatten()
    )
    print(
        f"Number of IDs with fewer than 35 articles to summarize: {np.sum(lengths < 35)}"
    )
    return filtered


def tokenize_and_count(sentence_df: pl.DataFrame):
    df = sentence_df.with_columns(
        [pl.col("sentence").apply(get_token_length).alias("num_tokens")]
    )  # , pl.col("sentence").apply(encode_sentences).alias("sentence_encoding")
    df = df.with_columns(pl.col("num_tokens").arr.sum().alias("total")).sort(
        "total", descending=True
    )
    print(
        f"Number of RNAs with fewer than 3072 total tokens: {df.filter(pl.col('total').lt(3072)).height}"
    )
    return df


if __name__ == "__main__":
    sentence_df = get_sentences()
    sentence_df = tokenize_and_count(sentence_df)
    print(sentence_df)

    sample_df = sentence_df.filter(pl.col("total").lt(3072)).select(
        ["job_id", "pmcid", "sentence"]
    )

    print(sample_df)
