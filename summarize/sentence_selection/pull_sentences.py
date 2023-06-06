"""
This script just orchestrates the execution of an SQL query against the given DB and saves the result to a JSON file.



The SQL query should return a table with the following columns:
- job_id
- result_id
- pmcid
- sentence

The result of the query will be available as a polars dataframe, and saved to JSON when running this as a script.

"""

import logging

import click
import polars as pl
import psycopg2
import psycopg2.extras


def pull_data_from_db(conn_str, query):
    """
    Just executes the pull using connextorx in the background. Should be directly writable to JSON, or useable as a dataframe.
    """
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(query)
    res = cur.fetchall()
    logging.info(f"Got {len(res)} results from database")
    df = pl.DataFrame(res)

    return df


@click.command()
@click.option("--conn_str", envvar="PGDATABASE", type=str)
@click.option("--query", type=click.Path(exists=True))
@click.option("--output_file", type=click.Path())
def main(conn_str, query, output_file):
    df = pull_data_from_db(conn_str, query)
    df.write_json(output_file)
