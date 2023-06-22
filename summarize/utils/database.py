import logging
import os
from collections import namedtuple

import polars as pl
import psycopg2
from psycopg2.extras import execute_values

Settings = namedtuple(
    "Settings",
    [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DATABASE",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "ENVIRONMENT",
    ],
)


def insert_rna_data(data_dict, conn_str, interactive=False, overwrite=False):
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    if interactive:
        num = cur.execute("select count(*) from litsumm_summaries").fetchone()
        if num > 0:
            choice = input(
                f"Data already exists in database! Would you like to (o)verwrite, (a)ppend, or (s)kip? ({num} rows)"
            )
            if choice == "o":
                cur.execute("truncate table litsumm_summaries")
            elif choice == "a":
                logging.warning("Appending data!")
            elif choice == "s":
                logging.warning("Skipping!")
                print("Skipping!")
                return
    else:
        if overwrite:
            logging.warning("Overwriting data in database!")
            cur.execute("truncate table litsumm_summaries")
        else:
            logging.warning(
                "Appending data! Things might get funky if inserting duplicate entity IDs!"
            )

    data = [
        (
            e["ent_id"],
            e["context"],
            e["summary"],
            e["cost"],
            e["total_tokens"],
            e["attempts"],
            e["problem_summary"],
            e["truthful"],
            e["consistency_check_result"],
        )
        for e in data_dict
    ]
    insert_query = "insert into litsumm_summaries (rna_id, context, summary, cost, total_tokens, attempts, problem_summary, truthful, consistency_check_result) values %s"
    execute_values(cur, insert_query, data, page_size=100)
    conn.commit()
    cur.close()
    conn.close()
    # df = pl.DataFrame(data_dict)
    # try:
    #     df.write_database("litscan_article_summaries", conn_str, if_exists="fail")
    # except:
    #     logging.warning("Data already exists in database!")
    #     if interactive:
    #         choice = input(
    #             "Data already exists in database! Would you like to (o)verwrite, (a)ppend, or (s)kip?"
    #         )
    #         if choice == "o":
    #             df.write_database(
    #                 "litscan_article_summaries", conn_str, if_exists="replace"
    #             )
    #         elif choice == "a":
    #             df.write_database(
    #                 "litscan_article_summaries", conn_str, if_exists="append"
    #             )
    #         elif choice == "s":
    #             logging.warning("Skipping!")
    #             print("Skipping!")
    #     else:
    #         logging.warning("Overwriting data in database!")
    #         df.write_database(
    #             "litscan_article_summaries", conn_str, if_exists="replace"
    #         )


def get_postgres_credentials(ENVIRONMENT):
    ENVIRONMENT = ENVIRONMENT.upper()

    if ENVIRONMENT == "DOCKER":
        return Settings(
            POSTGRES_HOST="database",  # database image from docker compose
            POSTGRES_PORT=5432,
            POSTGRES_DATABASE=os.getenv("LITSCAN_DB", "reference"),
            POSTGRES_USER=os.getenv("LITSCAN_USER", "docker"),
            POSTGRES_PASSWORD=os.getenv("LITSCAN_PASSWORD", "example"),
            ENVIRONMENT=ENVIRONMENT,
        )
    elif ENVIRONMENT == "LOCAL":
        return Settings(
            POSTGRES_HOST="localhost",
            POSTGRES_PORT=5432,
            POSTGRES_DATABASE="reference",
            POSTGRES_USER="docker",
            POSTGRES_PASSWORD="example",
            ENVIRONMENT=ENVIRONMENT,
        )
    elif ENVIRONMENT == "PRODUCTION":
        return Settings(
            POSTGRES_HOST=os.getenv("POSTGRES_HOST", "192.168.0.6"),
            POSTGRES_PORT=os.getenv("POSTGRES_PORT", 5432),
            POSTGRES_DATABASE=os.getenv("POSTGRES_DATABASE", "reference"),
            POSTGRES_USER=os.getenv("POSTGRES_USER", "docker"),
            POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD", "pass"),
            ENVIRONMENT=ENVIRONMENT,
        )
    elif ENVIRONMENT == "TEST":
        return Settings(
            POSTGRES_HOST="localhost",
            POSTGRES_PORT=5432,
            POSTGRES_DATABASE="test_reference",
            POSTGRES_USER="docker",
            POSTGRES_PASSWORD="example",
            ENVIRONMENT="LOCAL",
        )


if __name__ == "__main__":
    data = []
    for n in range(1000):
        data.append({"rna_id": "a" * n, "context": "b" * n, "summary": "c" * n})

    insert_rna_data(data)
