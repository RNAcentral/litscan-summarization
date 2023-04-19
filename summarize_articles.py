from llm_abstraction.models import get_model
from chains.summarization import get_summarizer_chain, get_reference_chain

from sentence_selection.pull_intro_sentences import get_sentences_for_summary
from utils.database import insert_rna_data
from utils.googledocs import create_summary_doc, create_id_link_spreadsheet
from utils.context import build_context
from utils.validation import validate_summary

import click
from pathlib import Path
import polars as pl
import logging

def generate_summary(model_name, rna_id, context, evaluate_truth=False, max_rescue_attempts=3):
    """
    Runs the LLM chains to produce a summary, first the summarizer chain, 
    then runs some checking for the correctness of references. If needed, 
    it will then run the reference addition chain a few times until the references
    are adequately inserted.

    Optionally, we can run a checking chain to see if the summary makes factual 
    statements supported by the context
    """
    summary_chain = get_summarizer_chain(get_model(model_name, {"presence_penalty":-2, "frequency_penalty":1}), verbose=True)
    reference_chain = get_reference_chain(get_model(model_name, {"presence_penalty": 0, "frequency_penalty":0}),  verbose=True)


    summary = summary_chain.run(rna_id=rna_id, context_str=context)
    print(summary)

    validation = validate_summary(summary, context)
    attempt = 1
    while not all(validation.values()):
        if attempt >= max_rescue_attempts:
            logging.warn(f"Unable to generate a good summary for {rna_id}. Returning what we have and giving up")
            return summary
        logging.warn("Summary auto validation failed! Running reference insertion chain to rescue...")
        summary = reference_chain.run(rna_id=rna_id, context_str=context, summary=summary)
        validation = validate_summary(summary, context)
        attempt += 1

    print(summary)
    return summary







@click.command()
@click.option("--context_output_dir", default="contexts", type=click.Path() )
@click.option("--summary_output_dir", default="summaries", type=click.Path())
@click.option("--cached_sentences", default="sentences.json", type=click.Path())
@click.option("--generation_limit", default=-1)
@click.option("--start_idx", default=0)
@click.option("--dry_run", default=False, is_flag=True)
def main(context_output_dir, summary_output_dir, cached_sentences, generation_limit, start_idx, dry_run):
    context_output_dir = Path(context_output_dir)
    context_output_dir.mkdir(parents=True, exist_ok=True)
    summary_output_dir = Path(summary_output_dir)
    summary_output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "chatGPT"# this will be a CLI option at some point

    data_for_db = []

    if Path(cached_sentences).exists():
        sentence_df = pl.read_json(cached_sentences)
    else:
        sentence_df = get_sentences_for_summary()
        sentence_df.write_json(cached_sentences)

    if dry_run :
            print("Not running by request, exiting early")
            return
    ids_done = 0
    for idx, row in enumerate(sentence_df.iter_rows(named=True)):
        if start_idx > idx:
            continue
        context = build_context(row['selected_sentences'], row['selected_pmcids'])
        with open(context_output_dir/f"{row['job_id']}.txt", 'w') as context_output:
            context_output.write(context)
        
        summary = generate_summary(model_name, row["job_id"], context)
        with open(summary_output_dir/f"{row['job_id']}.txt", 'w') as summary_output:
            summary_output.write(summary)
        ids_done += 1

        data_for_db.append({'rna_id': row['job_id'], "context": context, "summary":  summary})
        if generation_limit < 0:
            continue
        elif ids_done == generation_limit-1: 
            break

    ## Insert the results into my database
    insert_rna_data(data_for_db)

    ## Create the googledocs
    documents = {}
    for entry in data_for_db:
        documents[entry['rna_id']] = create_summary_doc(entry['rna_id'], entry['context'], entry['summary'], system_instruction+context_padding+revision_context)
        

    ## Create the spreadsheet
    create_id_link_spreadsheet(documents)
    

if __name__ == "__main__":
    main()