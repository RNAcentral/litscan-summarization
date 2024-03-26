import re

import click
import polars as pl

pmcid_pattern = re.compile(r"PMC\d+")


## This is bad and I should fix it
context_padding = (
    "As an experienced academic who ALWAYS provides references for each sentence you write, "
    "produce a summary from the text below, focusing on {ent_id} and using the references for each sentence. "
    "\n\n{context_str}\n\n"
    "The reference for each sentence in the text is given at the end of the sentence, enclosed by []. "
    "For example, the first sentence has the reference [{first_ref}]. "
    "Refrences should only be provided at the end of sentences, and MUST follow the style in the context. Do not list references at the end of the summary. "
    "You MUST provide at least one reference per sentence you produce. "
    "Use only the information in the context given above. Start your summary with a brief description of {ent_id}, noting its type. "
    "Use 200 words or less."
    "\nSummary:\n"
)

system_instruction = (
    "You are an experienced academic and always provide references for each sentence you write. "
    "You are a researcher who always answers in a factual and unbiased way. "
    "Provide at least one reference per sentence you produce."
)

system_instruction_veracity = (
    "You are an experienced academic who has been asked to fact check a summary. "
    "You will check the validity of claims made, and that the claims have appropriate references. "
    "When making your assertions, you will only use the provided context, and will not use external sources"
)
veracity_context = (
    "Here is a bullet point list of statements about the entity {ent_id}:\n"
    "{bullet_summary}\n\n"
    "The summary was derived from the following context:\n"
    "{original_context}\n"
    "For each statement, determine whether it is true or false, based on whether there is supporting evidence in the context. "
    "Make a determination for all statements, If a statement is false, explain why.\n\n"
)


def expand_prompt(entries):
    ent_id = entries["ent_id"]
    context = entries["context"]
    first_ref = pmcid_pattern.findall(context)
    formatted_prompt = context_padding.format(
        ent_id=ent_id, context_str=context, first_ref=first_ref
    )
    return {
        "system": system_instruction,
        "input": formatted_prompt,
        "output": entries["summary"],
    }


def expand_veracity(entries):
    ent_id = entries["ent_id"]
    context = entries["context"]
    summary = entries["summary"]
    bullet_summary = "- " + summary.replace(". ", "\n- ")

    formatted_prompt = veracity_context.format(
        ent_id=ent_id, bullet_summary=bullet_summary, original_context=context
    )
    return {
        "system": system_instruction_veracity,
        "input": formatted_prompt,
        "output": entries["consistency_check_result"],
    }


@click.command()
@click.argument("raw_parquet_path")
@click.argument("output_parquet_path")
def main(raw_parquet_path, output_parquet_path):

    raw = pl.read_parquet(raw_parquet_path)
    print(raw.filter(pl.col("attempts") == 2))

    summarization_step0 = (
        raw.with_columns(
            res=pl.struct(
                pl.col("ent_id"), pl.col("context"), pl.col("summary")
            ).map_elements(expand_prompt)
        )
        .unnest("res")
        .select(["system", "input", "output"])
    )

    veracity_step = (
        raw.with_columns(
            res=pl.struct(
                pl.col("ent_id"),
                pl.col("context"),
                pl.col("summary"),
                pl.col("consistency_check_result"),
            ).map_elements(expand_veracity)
        )
        .unnest("res")
        .select(["system", "input", "output"])
    )

    ft_dataset = summarization_step0.vstack(veracity_step)

    ft_dataset.write_parquet(output_parquet_path)

    pass


if __name__ == "__main__":
    main()
