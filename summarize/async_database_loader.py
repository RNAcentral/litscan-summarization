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
def main(conn_str, summary_data):
    summary_data = pl.read_json(summary_data)
    print(summary_data)
    insert_rna_data(summary_data.to_dicts(), conn_str)


if __name__ == "__main__":
    main()
