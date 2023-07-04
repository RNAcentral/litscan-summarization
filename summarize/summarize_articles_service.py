"""
Creates a daemon that polls the database for new articles to summarize.
We search for articles in litscan_database whose ids are not in
litscan_article_summaries. When we find some, we summarize them and
put the result into the summaries table.
"""
import logging
import os
from time import sleep

import polars as pl
import psycopg2

logging.getLogger().setLevel(logging.INFO)

from sentence_selection import get_sentences
from utils.context import build_context
from utils.database import get_postgres_credentials, insert_rna_data

from summaries import generate_summary


def poll_litscan_job():
    ENVIRONMENT = os.getenv("ENVIRONMENT", "LOCAL")
    credentials = get_postgres_credentials(ENVIRONMENT)
    conn_str = f"postgresql://{credentials.POSTGRES_USER}:{credentials.POSTGRES_PASSWORD}@{credentials.POSTGRES_HOST}/{credentials.POSTGRES_DATABASE}"
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()

    ## Query to get those IDs we haven't summarized yet, but which are finished
    query = """
    SELECT j.job_id
    FROM litscan_job j
    LEFT JOIN litsumm_summaries a ON j.job_id = a.rna_id
    WHERE a.rna_id IS NULL
    AND j.hit_count > 0;
    """

    while True:
        logging.info("Polling for new jobs...")
        cur.execute(query)  # .
        res = cur.fetchall()
        if len(res) > 0:
            ## convert the list of tuples to a list of ids - then re-tupleify later
            res = [r[0] for r in res]
            logging.info(f"Got {len(res)} new jobs to summarize!)")
            run_summary_job(res, conn_str)
        sleep(30)


def run_summary_job(job_ids, conn_str):
    """
    Read the polling query, which is slightly different from the cli one - it doesn't have a blacklist
    and has space for the tuple of IDs we got from polling the tables
    """
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    query_template = open("polling_query.sql", "r").read()
    placeholders = ",".join(["%s"] * len(job_ids))
    query = query_template.format(placeholders=placeholders)
    query = cur.mogrify(query, tuple(job_ids))

    logging.info("Pulling sentences...")
    sentence_df = get_sentences.for_summary(
        conn_str,
        query=query,
        device=os.getenv("DEVICE", "cpu:0"),
        limit=int(os.getenv("TOKEN_LIMIT", 2560)),
        cache=None,
    )
    if sentence_df is None:
        logging.info("No sentences to summarize!")
        return
    ## Filter out IDs with no sentences
    print(sentence_df)
    # sentence_df = sentence_df.filter(pl.col("selected_sentences ").list.lengths() > 0)

    if len(sentence_df) == 0:
        logging.info("No sentences to summarize!")
        return

    data_for_db = []
    for row in sentence_df.iter_rows(named=True):
        context = build_context(row["selected_sentences"], row["selected_pmcids"])
        (
            summary,
            cost,
            total_tokens,
            attempts,
            problem_summary,
            truthful,
            veracity_check_result,
        ) = generate_summary(
            os.getenv("MODEL_NAME", "chatGPT"),
            row["primary_id"],
            context,
            evaluate_truth=True,
            extra_args={},  ## TODO: How to handle local models that require this to have the weights path in
        )
        logging.info(
            f"Generated summary for {row['primary_id']}! Cost {cost}, total tokens {total_tokens}."
        )
        data_for_db.append(
            {
                "ent_id": row["primary_id"],
                "context": context,
                "summary": summary,
                "cost": cost,
                "total_tokens": total_tokens,
                "attempts": attempts,
                "problem_summary": problem_summary,
                "truthful": truthful,
                "consistency_check_result": veracity_check_result,
                "selection_method": row["method"],
            }
        )
    logging.info("Inserting all summaries into database...")
    insert_rna_data(data_for_db, conn_str)


if __name__ == "__main__":
    logging.info("Starting polling daemon...")
    poll_litscan_job()
