from pathlib import Path

import click
import polars as pl
from sentence_selection import get_sentences
from utils.context import build_context
from utils.database import insert_rna_data
from utils.googledocs import create_id_link_spreadsheet, create_summary_doc

from summaries import generate_summary


@click.command()
@click.option("--context_output_dir", default="contexts", type=click.Path())
@click.option("--summary_output_dir", default="summaries", type=click.Path())
@click.option("--veracity_output_dir", default="veracity_checks", type=click.Path())
@click.option("--cached_sentences", default="sentences.json", type=click.Path())
@click.option("--conn_str", envvar="PGDATABASE")
@click.option("--device", default="cpu:0")
@click.option("--query_file", default="query.sql")
@click.option("--token_limit", default=2560)
@click.option("--evaluate_truth", default=True, is_flag=True)
@click.option("--write_db", default=False, is_flag=True)
@click.option("--write_json", default=False, is_flag=True)
@click.option("--write_gdocs", default=False, is_flag=True)
@click.option("--generation_limit", default=-1)
@click.option("--start_idx", default=0)
@click.option("--dry_run", default=False, is_flag=True)
@click.option("--model_name", default="chatGPT")
@click.option("--model_path", default=None)
def main(
    context_output_dir,
    summary_output_dir,
    veracity_output_dir,
    cached_sentences,
    evaluate_truth,
    generation_limit,
    start_idx,
    dry_run,
    device,
    query_file,
    token_limit,
    conn_str,
    write_db,
    write_json,
    write_gdocs,
    model_name,
    model_path,
):
    context_output_dir = Path(context_output_dir)
    context_output_dir.mkdir(parents=True, exist_ok=True)
    summary_output_dir = Path(summary_output_dir)
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    veracity_output_dir = Path(veracity_output_dir)
    veracity_output_dir.mkdir(parents=True, exist_ok=True)

    if model_path is not None:
        extra_args = {"model_path": model_path}
    else:
        extra_args = {}

    data_for_db = []

    if Path(cached_sentences).exists():
        sentence_df = pl.read_json(cached_sentences)
    else:
        query = Path(query_file).read_text()
        sentence_df = get_sentences.for_summary(
            conn_str,
            query=query,
            device=device,
            limit=token_limit,
            cache=Path("raw_sentences.json"),
        )
        sentence_df.write_json(cached_sentences)

    if dry_run:
        print("Not running by request, exiting early")
        return
    ids_done = 0
    for idx, row in enumerate(sentence_df.iter_rows(named=True)):
        if start_idx > idx:
            continue
        context = build_context(row["selected_sentences"], row["selected_pmcids"])
        with open(
            context_output_dir / f"{row['primary_id']}.txt", "w"
        ) as context_output:
            context_output.write(context)

        (
            summary,
            cost,
            total_tokens,
            attempts,
            problem_summary,
            truthful,
            veracity_check_result,
        ) = generate_summary(
            model_name,
            row["primary_id"],
            context,
            evaluate_truth=evaluate_truth,
            extra_args=extra_args,
        )
        with open(
            summary_output_dir / f"{row['primary_id']}.txt", "w"
        ) as summary_output:
            summary_output.write(summary)

        with open(
            veracity_output_dir / f"{row['primary_id']}.txt", "w"
        ) as veracity_output:
            veracity_output.write(veracity_check_result)
        ids_done += 1

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
            }
        )
        if generation_limit < 0:
            continue
        elif ids_done == generation_limit:
            break

    if write_db:
        ## Insert the results into my database
        insert_rna_data(data_for_db, conn_str)

    if write_json:
        ## Write the results to parquet
        df = pl.DataFrame(data_for_db)
        df.write_ndjson("summary_data.parquet")

    if write_gdocs:
        ## Create the googledocs
        documents = {}
        for entry in data_for_db:
            documents[entry["ent_id"]] = create_summary_doc(
                entry["ent_id"],
                entry["context"],
                entry["summary"],
                system_instruction + context_padding + revision_context,
            )

        ## Create the spreadsheet
        create_id_link_spreadsheet(documents)


if __name__ == "__main__":
    main()
