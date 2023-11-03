"""
Loads context and summary into the database asynchronously,
i.e. when we run the summary tool at the CLI and end up
with a huge pile of contexts and summaries to load into the
database, this is the tool that does it.
"""

import os

import click
import polars as pl
from utils.database import insert_rna_data


@click.command()
@click.option("--conn_str", envvar="PGDATABASE")
@click.option("--summary_data", default="summary_data.json", type=click.Path())
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--check_fkey", is_flag=True, default=False)
def main(conn_str, summary_data, overwrite, check_fkey):
    if summary_data.endswith("ndjson"):
        summary_data = pl.read_ndjson(summary_data)
    elif summary_data.endswith("parquet"):
        summary_data = pl.read_parquet(summary_data)
    print(summary_data)
    # summary_data = summary_data.rename({"ent_id": "rna_id"})
    print(summary_data)
    insert_rna_data(
        summary_data.to_dicts(), conn_str, overwrite=overwrite, check_fk=check_fkey
    )


if __name__ == "__main__":
    main()
