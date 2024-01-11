import json

import click
import polars as pl


@click.command()
@click.option(
    "--group_a_parquet", help="A parquet file containing generations in group A"
)
@click.option(
    "--group_b_parquet", help="A parquet file containing generations in group B"
)
@click.option(
    "--join_key", default="ent_id", help="The key on which to join dataframes"
)
@click.option(
    "--prompt_key",
    default="ent_id",
    help="The column in which the prompt is kept. Should be the same across generations.",
)
@click.option(
    "--generation_key_a",
    default="summary",
    help="The column in which generations in group A are found",
)
@click.option(
    "--generation_key_b",
    default="summary",
    help="The column in which generations in group B are found",
)
@click.option(
    "--prompt_variable_name",
    default="prompt",
    help="The name of the variable in labelstudio that will be filled with the prompt",
)
@click.option(
    "--gen_a_variable_name",
    default="gen_a",
    help="The name of the variable in labelstudio that will be filled with generations from A",
)
@click.option(
    "--gen_b_variable_name",
    default="gen_b",
    help="The name of the variable in labelstudio that will be filled with generations from B",
)
@click.option(
    "--output_file",
    default="rlhf_input.json",
    help="Where we should write the output. Must be JSON",
)
def main(
    group_a_parquet,
    group_b_parquet,
    join_key,
    prompt_key,
    generation_key_a,
    generation_key_b,
    prompt_variable_name,
    gen_a_variable_name,
    gen_b_variable_name,
    output_file,
):
    """
    Load each parquet file, only grab the columns we care about.
    Rename generation columns with generation variable names
    Join the dataframes on the join key
    """

    ## Handle prompt key being the same as join key:
    selections_a = [join_key, generation_key_a]
    selections_b = [join_key, generation_key_b]

    if prompt_key != join_key:
        selections_a.append(prompt_key)  # should be same in both, so only select once

    ## set up renaming
    rename_a = {generation_key_a: gen_a_variable_name, prompt_key: prompt_variable_name}
    rename_b = {generation_key_b: gen_b_variable_name}

    # laxy load parquet, select then join
    df_a = pl.scan_parquet(group_a_parquet).select(selections_a).rename(rename_a)
    if prompt_key == join_key:
        df_a = df_a.with_columns(pl.col(prompt_variable_name).alias(join_key))
    df_b = pl.scan_parquet(group_b_parquet).select(selections_b).rename(rename_b)

    output_df = df_a.join(df_b, on=join_key).select(
        [prompt_variable_name, gen_a_variable_name, gen_b_variable_name]
    )

    ## This format is particular to labelstudio I think
    with open(output_file, "w") as output:
        output.write(json.dumps([{"data": r} for r in output_df.collect().to_dicts()]))


if __name__ == "__main__":
    main()
