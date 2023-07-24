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
import re
import time

import click
import polars as pl
import psycopg2
import psycopg2.extras
import requests
from psycopg2.extras import execute_values
from tqdm import tqdm


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


def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo


def get_sentence_for_one(conn_str, query_template, data):
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    primary_id = data["primary_id"]
    ids = data["ids_to_search"]

    if len(primary_id) > 1:
        logging.info(
            f"Got multiple primary IDs for {primary_id} - choosing the one with the most hits"
        )
        id_hits = []
        for pid in primary_id:
            cur.execute(
                "select hit_count from litscan_job where job_id = %s", (pid.lower(),)
            )
            res = cur.fetchone()
            if res is not None:
                id_hits.append(res["hit_count"])
        if len(id_hits) == 0:
            logging.info(
                "No hits found for any of the primary IDs - choosing the first one"
            )
            primary_id = primary_id[0]
        else:
            ## Use the primary ID with the most hits as primary
            primary_id = primary_id[id_hits.index(max(id_hits))]
    else:
        primary_id = primary_id[0]

    placeholders = ",".join(["%s"] * len(ids))
    query = query_template.format(placeholders=placeholders)
    query = cur.mogrify(query, tuple([i.lower() for i in ids]))
    cur.execute(query)
    res = cur.fetchall()
    result = {"pmcid": [], "sentence": []}
    if len(res) > 0:
        for hit in res:
            for alias in ids:
                if alias in hit["sentence"]:
                    hit["sentence"] = re.sub(
                        alias, primary_id, hit["sentence"], flags=re.IGNORECASE
                    )
            result["pmcid"].append(hit["pmcid"])
            result["sentence"].append(hit["sentence"])

    return result


def get_sentence_for_many(conn_str, data):
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    ## Create temp table to insert IDs into
    cur.execute("CREATE TABLE temp_ids (id varchar(100), urs_taxid text);")

    ## make the df huge
    data = (
        data.groupby("urs_taxid")
        .agg(pl.col("primary_id"), pl.col("aliases").flatten())
        .with_columns(ids_to_search=pl.col("aliases").list.concat(pl.col("primary_id")))
        .select(["urs_taxid", "ids_to_search"])
        .explode("ids_to_search")
        .unique()
    )

    ## Insert IDs into temp table
    ids = data.get_column("ids_to_search").to_list()
    urs_taxids = data.get_column("urs_taxid").to_list()

    execute_values(
        cur,
        "INSERT INTO temp_ids (id, urs_taxid) VALUES %s",
        [(i, j) for i, j in zip(ids, urs_taxids)],
    )

    conn.commit()

    ## Get sentences
    query = """
    select urs_taxid, lsa.pmcid, coalesce(lsbs.sentence, lsas.sentence) as sentence
    from litscan_result lsr
    join litscan_article lsa on lsa.pmcid = lsr.pmcid
    join litscan_body_sentence lsbs on lsbs.result_id = lsr.id
    left join litscan_abstract_sentence lsas on lsas.result_id = lsr.id
    join temp_ids on temp_ids.id = lsr.job_id


    and not lsa.retracted
    and not coalesce(lsbs.sentence, lsas.sentence) like '%found in an image%'

    """
    cur.execute(query)
    res = pl.DataFrame(cur.fetchall())

    ## WIP - we now need to do the grouping, making sure to maintain order
    ## Should be doable in polars, but we need to have a manageable number of IDs first
    # res.write_parquet("mirbase_intermediate.parquet")
    return res


def handle_sgd(conn_str, query):
    """
    at the moment we don't have good gene names for SGD, but they do have an api where we can get the gene names from the SGD ID
    """

    def get_gene_name(sgd_id):
        res = requests.get(f"https://yeastgenome.org/backend/locus/{sgd_id}")
        time.sleep(0.1)
        if res.status_code == 200:
            primary = res.json()["display_name"]
            alias_list = res.json()["aliases"]
            alias_ids = []
            for alias in alias_list:
                if alias["source"]["display_name"] == "SGD":
                    alias_ids.append(alias["display_name"])
            return {"primary_id": primary, "aliases": alias_ids}
        else:
            return {"primary_id": None, "aliases": []}

    sgd_initial = pull_data_from_db(conn_str, query)
    pbar = tqdm(total=len(sgd_initial), desc="Fetching SGD gene names...", colour="red")

    sgd_initial = sgd_initial.with_columns(
        res=pl.col("external_id").apply(w_pbar(pbar, get_gene_name))
    ).unnest("res")

    return sgd_initial


def handle_mirbase(conn_str, query):
    mirbase_initial = pull_data_from_db(conn_str, query)
    mirgenedb = (
        pl.scan_parquet("../mirgenedb_sentences.parquet")
        .select(["urs_taxid", "primary_id"])
        .with_columns(primary_id=pl.col("primary_id").list.first())
        .collect()
    )

    print(mirbase_initial)
    print(mirgenedb)

    combo = mirbase_initial.join(mirgenedb, on="primary_id", how="anti")
    print(combo)
    sent = get_sentence_for_many(conn_str, combo)
    initial_df = (
        pl.scan_csv("mirbase_sentence_data.csv", low_memory=True).head(10).collect()
    )
    print(initial_df)
    exit()


def pull_initial(conn_str, query, database=""):
    """
    Pulls the initial data from the database. This is a bit different for SGD, so we have a separate function for that.
    """
    if database == "sgd":
        return handle_sgd(conn_str, query)
    elif database == "mirbase":
        return handle_mirbase(conn_str, query)
    else:
        return pull_data_from_db(conn_str, query)


def per_database_pull_sentences(
    conn_str, initial_df, sentence_template_query, pull_initial=True
):
    sentence_template_query = open(sentence_template_query, "r").read()
    grouped_on_urs = initial_df.groupby("urs_taxid").agg(
        pl.col("primary"), pl.col("aliases").flatten()
    )
    grouped_on_urs = grouped_on_urs.with_columns(
        ids_to_search=pl.col("primary")
        .list.concat(pl.col("aliases"))
        .list.unique()
        .list.eval(pl.element().filter(pl.element() != ""))
    ).sort("urs_taxid")
    grouped_on_urs.drop_nulls()

    pbar = tqdm(total=len(grouped_on_urs), desc="Fetching sentences...", colour="green")

    grouped_on_urs = grouped_on_urs.with_columns(
        result=pl.struct(pl.col("ids_to_search"), pl.col("primary_id")).apply(
            w_pbar(
                pbar,
                lambda x: get_sentence_for_one(conn_str, sentence_template_query, x),
            )
        )
    ).unnest("result")

    grouped_on_urs = grouped_on_urs.filter(pl.col("pmcid").list.lengths() > 0)

    return grouped_on_urs


@click.command()
@click.option("--conn_str", envvar="PGDATABASE", type=str)
@click.option("--query", type=click.Path(exists=True))
@click.option("--output_file", type=click.Path())
@click.option("--duplicates_file", type=click.Path())
@click.option("--database", type=str, default="")
def main(conn_str, query, output_file, duplicates_file, database):
    query = open(query, "r").read()
    initial = pull_initial(conn_str, query, database=database)

    ## Filter out non-specific miRNAs
    initial = initial.with_columns(
        aliases=pl.col("aliases")
        .list.eval(
            pl.element().filter(
                pl.when(pl.element().str.contains("mir"))
                .then(pl.element().str.count_match("-").gt(1))
                .otherwise(pl.lit(True))
            )
        )
        .list.eval(
            pl.element().filter(
                pl.when(pl.element().str.contains("iab"))
                .then(pl.element().str.count_match("-").gt(1))
                .otherwise(pl.lit(True))
            )
        )
        .list.eval(
            pl.element().filter(
                pl.when(pl.element().str.contains("let"))
                .then(pl.element().str.count_match("-").gt(1))
                .otherwise(pl.lit(True))
            )
        )
    )

    df, duplicates = per_database_pull_sentences(
        conn_str, initial, "../queries/get_sentences_per_urs.sql"
    )

    df.write_parquet(output_file)
    duplicates.write_csv(duplicates_file)


if __name__ == "__main__":
    main()
