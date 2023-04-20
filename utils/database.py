import os

import psycopg2
from psycopg2.extras import execute_values


def insert_rna_data(data_dict):
    conn = psycopg2.connect(os.getenv("SUMMDATABASE"))
    cur = conn.cursor()
    data = [(e["rna_id"], e["context"], e["summary"]) for e in data_dict]
    insert_query = (
        "insert into litscan_article_summaries (rna_id, context, summary) values %s"
    )
    execute_values(cur, insert_query, data, page_size=100)
    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    data = []
    for n in range(1000):
        data.append({"rna_id": "a" * n, "context": "b" * n, "summary": "c" * n})

    insert_rna_data(data)
