from pathlib import Path

import click
import polars as pl
from sentence_selection import get_sentences
from tqdm import tqdm
from utils.context import build_context
from utils.database import insert_rna_data
from utils.googledocs import create_id_link_spreadsheet, create_summary_doc

from summaries import generate_summary


def write_output(data_for_db, output_basename, write_json, write_parquet):
    df = pl.DataFrame(data_for_db)
    if write_json:
        ## Write the results to ndjson
        output_loc = Path(f"{output_basename}.ndjson")
        if output_loc.exists():
            existing = pl.read_ndjson(output_loc)
            df = existing.vstack(df).unique()
        df.write_ndjson(f"{output_basename}.ndjson")

    if write_parquet:
        ## Write the results to parquet
        output_loc = Path(f"{output_basename}.parquet")
        if output_loc.exists():
            existing = pl.read_parquet(output_loc)
            df = existing.vstack(df).unique()
        df.write_parquet(f"{output_basename}.parquet")


@click.command()
@click.option("--context_output_dir", default="contexts", type=click.Path())
@click.option("--summary_output_dir", default="summaries", type=click.Path())
@click.option("--veracity_output_dir", default="veracity_checks", type=click.Path())
@click.option("--cached_sentences", default="sentences.json", type=click.Path())
@click.option("--conn_str", envvar="PGDATABASE")
@click.option("--evaluate_truth", default=True, is_flag=True)
@click.option("--write_db", default=False, is_flag=True)
@click.option("--write_json", default=False, is_flag=True)
@click.option("--write_parquet", default=False, is_flag=True)
@click.option("--generation_limit", default=-1)
@click.option("--start_idx", default=0)
@click.option("--dry_run", default=False, is_flag=True)
@click.option("--output_basename", default="summary_data")
@click.option("--checkpoint_interval", default=50)
@click.option("--model_name", default="chatGPT")
@click.option("--model_path", default=None)
@click.option("--verbosity", default=False, is_flag=True)
def main(
    context_output_dir,
    summary_output_dir,
    veracity_output_dir,
    cached_sentences,
    evaluate_truth,
    generation_limit,
    start_idx,
    dry_run,
    output_basename,
    checkpoint_interval,
    conn_str,
    write_db,
    write_json,
    write_parquet,
    model_name,
    model_path,
    verbosity,
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
        if cached_sentences.endswith(".parquet") or cached_sentences.endswith(".pq"):
            sentence_df = pl.read_parquet(cached_sentences)
        elif cached_sentences.endswith(".json"):
            sentence_df = pl.read_json(cached_sentences)
    else:
        print("The path to the prepared sentences doesn't seem to exist..?")

    if dry_run:
        print("Not running by request, exiting early")
        return
    ids_done = 0
    for idx, row in tqdm(enumerate(sentence_df.iter_rows(named=True))):
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
            rescue_prompts,
            problem_summary,
            truthful,
            veracity_check_result,
        ) = generate_summary(
            model_name,
            row["primary_id"],
            context,
            evaluate_truth=evaluate_truth,
            extra_args=extra_args,
            verbose=verbosity,
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
                "rescue_prompts": rescue_prompts,
                "problem_summary": problem_summary,
                "truthful": truthful,
                "consistency_check_result": veracity_check_result,
                "selection_method": row["method"],
                "urs_taxid": row["urs_taxid"],
            }
        )
        if generation_limit < 0:
            continue
        elif ids_done == generation_limit:
            break

        ## Checkpoint every N RNAs, default to 50 which should be about 20 minutes
        if (
            len(sentence_df) > checkpoint_interval
            and ids_done % checkpoint_interval == 0
        ):
            write_output(
                data_for_db,
                output_basename,
                write_json,
                write_parquet,
            )

    write_output(data_for_db, output_basename, write_json, write_parquet)
    if write_db:
        ## Insert the results into my database
        insert_rna_data(data_for_db, conn_str)


if __name__ == "__main__":
    main()
