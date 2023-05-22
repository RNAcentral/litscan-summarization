import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")


import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

import sentence_selection.with_location

QUERY = """select result_id, (array_agg( DISTINCT lsa.pmcid))[1] as pmcid,
				(array_agg( DISTINCT lsr.job_id))[1] as job_id,
				(array_agg(sentence))[1] as sentence
from embassy_rw.litscan_body_sentence lsb
join embassy_rw.litscan_result lsr on lsr.id = lsb.result_id
join embassy_rw.litscan_database lsdb on lsdb.job_id = lsr.job_id
join embassy_rw.litscan_article lsa on lsa.pmcid = lsr.pmcid
where name in ('pombase', 'hgnc', 'wormbase', 'mirbase', 'snodb', 'tair', 'sgd', 'pdbe', 'genecards', 'gtrnadb', 'mirgenedb', 'refseq', 'rfam', 'zfin' )
and retracted = false
and lsr.job_id not in ('12s', '12s rrna', '12 s rrna',
                       '13a', '16s', '16s rna', 'rrna', 'e3', 'e2',
                       '16srrna', '16s rrna', 'bdnf', 'nmr',
                       '2a-1', '2b-2', '45s pre-rrna', '7sk',
                       '7sk rna', '7sk snrna', '7slrna', 'rnai',
                       '7sl rna', 'trna', 'snrna', 'mpa', 'msa', 'rns', 'tran',
                       'mir-21', 'mir-155')
group by result_id

having cardinality(array_agg(lsb.id)) > 2 and cardinality(array_agg(DISTINCT lsr.job_id)) = 1
"""


def get_token_length(sentences):
    return [len(enc.encode(s)) for s in sentences]


def iterative_sentence_selector(row, token_limit=3072):
    sentences = row["sentence"]
    rna_id = row["job_id"]
    ## If we don't have enough, return all
    if sum(get_token_length(sentences)) <= token_limit:
        print(f"Few tokens for {rna_id}, using all sentences")
        return list(range(0, len(sentences)))
    sentences = np.array(sentences)

    ## Try to reduce the number of sentences that need to be encoded by clustering and taking exemplars

    encodings = torch.vstack(
        [
            model.encode(s, convert_to_tensor=True, normalize_embeddings=True)
            for s in sentences
        ]
    )
    communities = util.community_detection(
        encodings, min_community_size=5
    )  ## Tune min size to strike a balance between number of clusters and diversity

    selected_sentence_idxs = [
        c.pop(0) for c in communities
    ]  ## index 0 is the central point in the cluster
    encodings = encodings.cpu().numpy()
    selected_sentences_clustering = sentences[selected_sentence_idxs]
    selected_embeddings_clustered = encodings[selected_sentence_idxs]
    ## Check total length of selection now
    lengths = get_token_length(selected_sentences_clustering)
    ## If there aren't too many clusters, this will be true, and we can select from them until we run out of tokens
    if sum(lengths) <= token_limit:
        print(f"Sampling communities for {rna_id}, until token limit")
        ## round-robin grabbing of sentences until we hit the limit
        com_idx = 0
        while sum(get_token_length(selected_sentences_clustering)) < token_limit:
            if len(communities[com_idx]) > 0:
                selected_sentence_idxs.append(
                    communities[com_idx].pop(0)
                )  ## Should repeatedly remove the first sentence in each community
                selected_sentences_clustering = sentences[selected_sentence_idxs]
            else:
                break
            com_idx += 1
            com_idx %= len(communities)

        ## pop the last one, since by definition we went over by including it
        selected_sentence_idxs.pop()

        return selected_sentence_idxs

    print(f"{rna_id} has too many clusters to use round-robin selection")
    print(
        f"Using greedy selection algorithm for {rna_id}. Only works on cluster centres."
    )

    ## If we're here, there are still too many tokens in the selection. Now we need to optimize for diversity and token count
    ## Use a greedy algorithm on the cluster centres, start with the first because it should be the largest cluster
    selected_sentences_greedy_idxs = [selected_sentence_idxs[0]]
    selected_sentences_greedy = [selected_sentences_clustering[0]]
    selected_embeddings_greedy = [selected_embeddings_clustered[0]]
    total_tokens = sum(get_token_length(selected_sentences_greedy))
    while total_tokens < token_limit:
        for selected_embedding in selected_embeddings_greedy:
            distances = []
            cost = []
            idx_to_copy = []
            for idx, embedding in enumerate(selected_embeddings_clustered):
                if idx in selected_sentences_greedy_idxs:
                    continue
                distances.append(util.pairwise_dot_score(selected_embedding, embedding))
                cost.append(distances[-1] * lengths[idx])
                idx_to_copy.append(idx)

            next_selection = idx_to_copy[np.argmin(cost)]
            selected_sentences_greedy_idxs.append(next_selection)
            selected_sentences_greedy.append(
                selected_sentences_clustering[next_selection]
            )
            ## Check we aren't about to run out of tokens
            total_tokens = sum(get_token_length(selected_sentences_greedy))
            if total_tokens >= token_limit:
                selected_sentences_greedy.pop()
                selected_sentences_greedy_idxs.pop()
                break
    return selected_sentences_greedy_idxs


def select_sentences(row):
    sentences = np.array(row["sentence"])
    indices = row["selected_sentence_idxs"]
    return list(sentences[indices])


def select_pmcids(row):
    pmcids = np.array(row["pmcid"])
    indices = row["selected_sentence_idxs"]
    return list(pmcids[indices])


def sample_sentences(sentences: pl.DataFrame):
    df = sentences.with_columns(
        pl.struct(["sentence", "job_id"])
        .apply(iterative_sentence_selector)
        .alias("selected_sentence_idxs")
    ).filter(pl.col("selected_sentence_idxs").is_not_null())
    df = df.with_columns(
        [
            pl.struct(["sentence", "selected_sentence_idxs", "job_id"])
            .apply(select_sentences)
            .alias("selected_sentences")
        ]
    )
    df = df.with_columns(
        [
            pl.struct(["pmcid", "selected_sentence_idxs"])
            .apply(select_pmcids)
            .alias("selected_pmcids")
        ]
    )
    return df


def get_sentences():
    conn_str = os.getenv("PGDATABASE")
    df = pl.read_database(QUERY, conn_str)

    filtered = (
        df.groupby(["job_id"])
        .agg(pl.col("*"))
        .filter(pl.col("result_id").arr.lengths() > 1)
    )

    lengths = filtered.select(pl.col("result_id").arr.lengths()).to_numpy().flatten()
    print(
        f"Number of IDs with fewer than 35 articles to summarize: {np.sum(lengths < 35)}"
    )
    return filtered


def tokenize_and_count(sentence_df: pl.DataFrame):
    df = sentence_df.with_columns(
        [pl.col("sentence").apply(get_token_length).alias("num_tokens")]
    )  # , pl.col("sentence").apply(encode_sentences).alias("sentence_encoding")
    df = df.with_columns(
        [
            pl.col("num_tokens").arr.sum().alias("total"),
            pl.col("pmcid").arr.lengths().alias("num_articles"),
        ]
    ).sort("total", descending=True)
    print(
        f"Number of RNAs with fewer than 3072 total tokens: {df.filter(pl.col('total').lt(3072)).height}"
    )
    return df


def get_sentences_for_summary(method) -> pl.DataFrame:
    if method == "topic":
        if Path("all_available_sentences.json").exists():
            sentence_df = pl.read_json("all_available_sentences.json")
        else:
            sentence_df = get_sentences()
            sentence_df = tokenize_and_count(sentence_df)
            sentence_df.write_json("all_available_sentences.json")
        print(sentence_df)
        sample_df = sample_sentences(sentence_df)

        return sample_df.select(["job_id", "selected_pmcids", "selected_sentences"])
    elif method == "intro":
        sentence_df = with_location.get_sentences()
        sentence_df = with_location.tokenize_and_count(sentence_df)
        sentence_df.write_json("all_available_sentences_with_location.json")

        sample_df = sentence_df.filter(pl.col("total").lt(3072)).select(
            ["job_id", "selected_pmcids", "selected_sentences"]
        )

        return sample_df
