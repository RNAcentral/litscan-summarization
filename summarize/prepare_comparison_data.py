import json
from random import shuffle

import click
import numpy as np
import polars as pl


def randomise_side(row, gen_a_name, gen_b_name):
    """
    Randomise which side each generation is on, but keep track of it
    Also randomises the source column with the same shuffle
    Sadly, requires numpy
    """

    idxs = [0, 1]
    shuffle(idxs)

    ret_gen = np.array(row["summary"])[idxs]
    ret_src = np.array(row["source"])[idxs]
    return {
        gen_a_name: ret_gen[0],
        gen_b_name: ret_gen[1],
        "left": ret_src[0],
        "right": ret_src[1],
    }


@click.command()
@click.option(
    "--group_a_parquet", help="A parquet file containing generations in group A"
)
@click.option(
    "--group_b_parquet", help="A parquet file containing generations in group B"
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
    "--source_name_a",
    default="GPT3.5",
    help="The name of the model used to generate dataset A",
)
@click.option(
    "--generation_key_b",
    default="summary",
    help="The column in which generations in group B are found",
)
@click.option(
    "--source_name_b",
    default="GPT4",
    help="The name of the model used to generate dataset B",
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
    prompt_key,
    generation_key_a,
    source_name_a,
    generation_key_b,
    source_name_b,
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

    # load parquet, select then add source
    df_a = (
        pl.scan_parquet(group_a_parquet)
        .select([prompt_key, generation_key_a])
        .with_columns(source=pl.lit(source_name_a))
        .collect()
    )
    df_b = (
        pl.scan_parquet(group_b_parquet)
        .select([prompt_key, generation_key_b])
        .with_columns(source=pl.lit(source_name_b))
        .collect()
    )

    ## We want the generation to be called the same thig, so rename b to a if it isn't already
    if generation_key_a != generation_key_b:
        df_b = df_b.rename({generation_key_b, generation_key_a})

    inter = df_a.vstack(df_b)
    inter = inter.groupby(prompt_key).agg([pl.col(generation_key_a), pl.col("source")])
    inter = inter.with_columns(
        res=pl.struct([generation_key_a, "source"]).apply(
            lambda x: randomise_side(x, gen_a_variable_name, gen_b_variable_name)
        )
    ).unnest("res")
    inter = inter.select(
        [prompt_key, gen_a_variable_name, gen_b_variable_name, "left", "right"]
    ).rename({prompt_key: prompt_variable_name})

    print(inter)

    ## This format is particular to labelstudio I think
    with open(output_file, "w") as output:
        output.write(json.dumps([{"data": r} for r in inter.to_dicts()]))


if __name__ == "__main__":
    main()
