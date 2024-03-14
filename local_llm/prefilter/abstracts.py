import os
from pathlib import Path

import click
import lmql
import polars as pl
import psycopg2
from tqdm import tqdm
from transformers import AutoTokenizer


def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo


def write_output(evaluated, output):
    df = pl.DataFrame(evaluated)

    output_loc = Path(f"{output}")
    if output_loc.exists():
        existing = pl.read_parquet(output_loc)
        df = existing.vstack(df).unique("pmcid")
    df.write_parquet(output)


@lmql.query(decoder="sample", n=1, temperature=0.1, max_len=4096)
def classify_abstract(abstract):
    '''lmql
    """
    You are an academic with experience in molecular biology who has been
    asked to help classify research papers based on their abstracts.
    Below is an abstract from a paper. We need to filter out papers that
    are not about non-coding RNA. You will briefly analyse the abstract
    before classifying it, giving your reasoning why the abstract is
    relevant to ncRNA or not.

    Abstract: {abstract}

    Based on this abstract, does the paper contain information about
    non-coding RNA?
    [ANALYSIS]
    """ where STOPS_AT(ANALYSIS, "\n\n")

    """Therefore, in the context of ncRNA this paper is
    [CLS]""" distribution CLS in ["relevant", "not relevant"]

    '''


def classify_abstracts_df(abstract_text, model):
    """
    Basically a wrapper function to apply the LLM classification across a
    dataframe
    """

    r = classify_abstract(abstract_text, model=model)
    return r.variables["P(CLS)"][0][1]


QUERY = """select lsr.pmcid, lsr.job_id, abstract from litscan_result lsr
join litscan_database lsdb
	on lsdb.job_id = lsr.job_id
join litscan_article lsa
	on lsa.pmcid = lsr.pmcid
where lsdb.name = %s"""

PGDATABASE = os.getenv("PGDATABASE")


@click.command()
@click.argument("abstracts")
@click.argument("output")
@click.option(
    "--model_path", default="/Users/agreen/LLMs/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
)
@click.option("--database", default="flybase")
@click.option("--ngl", default=1)
@click.option("--chunks", default=4)
@click.option("--gpu_number", default=0)
@click.option("--checkpoint_frequency", default=50)
def main(
    abstracts,
    output,
    model_path,
    database,
    ngl,
    chunks,
    gpu_number,
    checkpoint_frequency,
):
    if abstracts == "fetch":
        conn = psycopg2.connect(PGDATABASE)
        cur = conn.cursor()
        completed_query = cur.mogrify(QUERY, (database,))
        print(completed_query)
        abstracts = (
            pl.read_database(completed_query, conn)
            .unique("pmcid")
            .with_row_count(name="index")
        )
        n_c = abstracts.height // chunks
        pieces = [
            abstracts.filter(pl.col("index").is_between(a * n_c, (a + 1) * n_c))
            for a in range(chunks)
        ]
        for n, piece in enumerate(pieces):
            piece.write_parquet(f"{output}_{n}.pq")
        exit()
    else:
        abstracts = pl.read_parquet(abstracts)

    ## Check for a previous checkpoint and resume where we left off
    if Path(output).exists():
        checkpointed = pl.read_parquet(output)
        abstracts = abstracts.join(checkpointed, on="pmcid", how="anti")
        print(f"Resuming from checkpoint, {abstracts.height} to go")

    ## Set environment with supplied GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # abstract_text = """RNase P RNA (RPR), the catalytic subunit of the essential RNase P ribonucleoprotein, removes the 5' leader from precursor tRNAs. The ancestral eukaryotic RPR is a Pol III transcript generated with mature termini. In the branch of the arthropod lineage that led to the insects and crustaceans, however, a new allele arose in which RPR is embedded in an intron of a Pol II transcript and requires processing from intron sequences for maturation. We demonstrate here that the Drosophila intronic-RPR precursor is trimmed to the mature form by the ubiquitous nuclease Rat1/Xrn2 (5') and the RNA exosome (3'). Processing is regulated by a subset of RNase P proteins (Rpps) that protects the nascent RPR from degradation, the typical fate of excised introns. Our results indicate that the biogenesis of RPR in vivo entails interaction of Rpps with the nascent RNA to form the RNase P holoenzyme and suggests that a new pathway arose in arthropods by coopting ancient mechanisms common to processing of other noncoding RNAs."""
    print(abstracts)

    model = lmql.model(
        f"local:llama.cpp:{model_path}",
        tokenizer="mistralai/Mixtral-8x7B-Instruct-v0.1",
        n_gpu_layers=ngl,
        n_ctx=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

    ## filter out abstracts with zero or too many tokens
    abstracts = abstracts.with_columns(
        n_tokens=pl.col("abstract").apply(lambda x: len(tokenizer.encode(x)))
    )
    abstracts = abstracts.filter(pl.col("n_tokens").is_between(100, 3000))

    # r = classify_abstract(abstract_text, model=model,  output_writer=lmql.printing)
    # print(r)
    # exit()

    classified_pmcids = []
    relevant_probability = []
    for idx, paper in tqdm(
        enumerate(abstracts.iter_rows(named=True)), total=abstracts.height
    ):
        abstract = paper["abstract"]
        pmcid = paper["pmcid"]

        rel_prob = classify_abstracts_df(abstract, model)

        classified_pmcids.append(pmcid)
        relevant_probability.append(rel_prob)

        if idx % checkpoint_frequency == 0:
            evaluated = {
                "pmcid": classified_pmcids,
                "relevance_probability": relevant_probability,
            }
            write_output(evaluated, output)
            ## Reset accumulating lists
            classified_pmcids = []
            relevant_probability = []

    ## Finished, so write everything else
    evaluated = {
        "pmcid": classified_pmcids,
        "relevance_probability": relevant_probability,
    }
    write_output(evaluated, output)


if __name__ == "__main__":
    main()
