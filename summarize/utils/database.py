import logging

import polars as pl


def insert_rna_data(data_dict, conn_str, interactive=False):
    df = pl.DataFrame(data_dict)
    try:
        df.write_database("litscan_article_summaries", conn_str, if_exists="fail")
    except:
        logging.warning("Data already exists in database!")
        if interactive:
            choice = input(
                "Data already exists in database! Would you like to (o)verwrite, (a)ppend, or (s)kip?"
            )
            if choice == "o":
                df.write_database(
                    "litscan_article_summaries", conn_str, if_exists="replace"
                )
            elif choice == "a":
                df.write_database(
                    "litscan_article_summaries", conn_str, if_exists="append"
                )
            elif choice == "s":
                logging.warning("Skipping!")
                print("Skipping!")
        else:
            logging.warning("Overwriting data in database!")
            df.write_database(
                "litscan_article_summaries", conn_str, if_exists="replace"
            )


if __name__ == "__main__":
    data = []
    for n in range(1000):
        data.append({"rna_id": "a" * n, "context": "b" * n, "summary": "c" * n})

    insert_rna_data(data)
