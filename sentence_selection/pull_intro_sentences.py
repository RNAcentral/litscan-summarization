import psycopg2
from psycopg2.extras import RealDictCursor
import polars as pl
import os
import matplotlib.pyplot as plt
import numpy as np
import typing as ty
import torch
from scipy.optimize import linear_sum_assignment

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="mps")


import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

QUERY = """select result_id, (array_agg( DISTINCT lsa.pmcid))[1] as pmcid,
				(array_agg( DISTINCT lsr.job_id))[1] as job_id, 
				(array_agg(sentence))[1] as sentence
from embassy_rw.litscan_body_sentence lsb
join embassy_rw.litscan_result lsr on lsr.id = lsb.result_id
join embassy_rw.litscan_database lsdb on lsdb.job_id = lsr.job_id
join embassy_rw.litscan_article lsa on lsa.pmcid = lsr.pmcid
where name in ('pombase', 'hgnc', 'wormbase', 'mirbase')
and retracted = false
and lsr.job_id not in ('12s', '12s rrna', '12 s rrna', 
                       '13a', '16s', '16s rna', 
                       '16srrna', '16s rrna', 
                       '2a-1', '2b-2', '45s pre-rrna', '7sk',
                       '7sk rna', '7sk snrna', '7slrna',
                       '7sl rna', 'trna', 'snrna', 'mpa', 'msa', 'rns', 'tran')
group by result_id

having cardinality(array_agg(lsb.id)) > 2 and cardinality(array_agg(DISTINCT lsr.job_id)) = 1
"""


def get_token_length(sentences):
    return [len(enc.encode(s)) for s in sentences]


def iterative_sentence_selector(row, token_limit=2048):
    sentences = row['sentence']
    rna_id = row['job_id']
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
    selected_sentences_clustering = sentences[selected_sentence_idxs]
    selected_embeddings_clustered = encodings[selected_sentence_idxs]
    ## Check total length of selection now
    lengths = get_token_length(selected_sentences_clustering)
    if sum(lengths) <= token_limit:
        if len(communities) > 2:
            print(f"More than 2 communities for {rna_id}, using their centroids")
            return selected_sentence_idxs
        else:
            print(f"Too few communities for {rna_id}, sampling from them in order until token limit")
            ## round-robin grabbing of sentences until we hit the limit
            com_idx = 0
            while sum(get_token_length(selected_sentences_clustering)) < token_limit:
                if len(communities[com_idx]) > 0:
                    selected_sentence_idxs.append(communities[com_idx].pop(0)) ## Should repeatedly remove the first sentence in each community
                    selected_sentences_clustering = sentences[selected_sentence_idxs]
                else:
                    break
                com_idx += 1
                com_idx %= len(communities)

            ## pop the last one, since by definition we went over by including it
            selected_sentence_idxs.pop()

            return selected_sentence_idxs        


    print(f"Using greedy selection algorithm for {rna_id}. Only works on cluster centres.")

    ## If we're here, there are still too many tokens in the selection. Now we need to optimize for diversity and token count
    ## Use a greedy algorithm on the cluster centres, start with the first because it should be the largest cluster
    selected_sentences_greedy_idxs = [0]
    selected_sentences_greedy = [selected_sentences_clustering[0]]
    total_tokens = sum(get_token_length(selected_sentences_greedy))
    while total_tokens < token_limit:
        for selected_sentence in selected_sentences_greedy:
            distances = []
            cost = []
            idx_to_copy = []
            for idx, sentence in enumerate(selected_sentences_clustering):
                if idx in selected_sentences_greedy_idxs:
                    continue
                distances.append(
                    util.pairwise_dot_score(
                        model.encode(selected_sentence), model.encode(sentence)
                    )
                )
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
    conn = psycopg2.connect(os.getenv("PGDATABASE"))
    dict_cur = conn.cursor(cursor_factory=RealDictCursor)
    dict_cur.execute(QUERY)
    ret = dict_cur.fetchall()
    df = pl.DataFrame(ret)

    filtered = (
        df.groupby(["job_id"])
        .agg(pl.col("*"))
        .filter(pl.col("result_id").arr.lengths() > 1)
    )

    lengths = filtered.select(pl.col("result_id").arr.lengths()).to_numpy().flatten()
    print(
        f"Number of IDs with fewer than 35 articles to summarize: {np.sum(lengths < 25)}"
    )
    return filtered


def tokenize_and_count(sentence_df: pl.DataFrame):
    df = sentence_df.with_columns(
        [pl.col("sentence").apply(get_token_length).alias("num_tokens")]
    )  # , pl.col("sentence").apply(encode_sentences).alias("sentence_encoding")
    df = df.with_columns(pl.col("num_tokens").arr.sum().alias("total")).sort(
        "total", descending=True
    )
    print(
        f"Number of RNAs with fewer than 2048 total tokens: {df.filter(pl.col('total').lt(2048)).height}"
    )
    return df


def get_sentences_for_summary() -> pl.DataFrame:
    sentence_df = get_sentences()
    sentence_df = tokenize_and_count(sentence_df)
    sentence_df.write_json("all_available_sentences.json")
    
    sample_df = sample_sentences(sentence_df)

    return sample_df.select(["job_id", "selected_pmcids", "selected_sentences"])
