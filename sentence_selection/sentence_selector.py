"""
Uses output of topic modelling to select sentences for each ID

Input: JSON file containing sentences grouped by ID, and topic modelled
Output: JSON file containing sentences selected for each ID

"""

import logging

import numpy as np
import polars as pl
from sentence_transformers import util

from sentence_selection.topic_modelling import run_topic_modelling
from sentence_selection.utils import get_token_length


def iterative_sentence_selector(row, model, token_limit=3072):
    """
    Applies some heuristics to select sentence indices for each ID

    Basic - if the sum of all available sentences' tokens is less than the token limit, return all sentences
    If not, use topic modelling result. Take cluster centres, and greedily select sentences until we hit the token limit
    Greedy selection uses the embedding of each sentence to calculate the most distinct sentence left in the cluster

    Input: row from dataframe containing sentences and ID
    Input: model - sentence transformer model
    Input: token_limit - maximum number of tokens to use for each ID

    Output: list of sentence indices to use for each ID

    """
    ent_id = row["job_id"]
    row = pl.DataFrame(row)
    sentences = row.get_column("sentence").to_list()
    pmcids = row.get_column("pmcid").to_list()

    ## If we don't have enough, return all
    if sum(get_token_length(sentences)) <= token_limit:
        logging.info(f"Few tokens for {ent_id}, using all sentences")
        return {
            "selected_sentences": sentences,
            "selected_pmcids": pmcids,
        }

    ## If we have too many, use topic modelling
    logging.info(f"Too many tokens for {ent_id}, using topic modelling")
    row = run_topic_modelling(row, model)

    sentences = np.array(sentences)
    pmcids = np.array(pmcids)

    ## Try to reduce the number of sentences that need to be encoded by clustering and taking exemplars

    embeddings = (
        model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
        .cpu()
        .numpy()
    )
    labels = row.get_column("sentence_labels").to_numpy()
    communities = np.unique(sorted(labels))

    ## Select a starting sentence per cluster
    selected_sentences = [sentences[labels == i][0] for i in np.unique(labels) if i > 0]
    selected_pmcids = [pmcids[labels == i][0] for i in np.unique(labels) if i > 0]

    label_index_lookup = {i: 1 for i in np.unique(labels) if i > 0}
    ## If there aren't too many clusters, this will be true, and we can select from them until we run out of tokens
    if sum(get_token_length(selected_sentences)) <= token_limit:
        logging.info(f"Sampling communities for {ent_id}, until token limit")
        ## round-robin grabbing of sentences until we hit the limit
        com_idx = 1  ## The -1 community is "noise", so we start at 1
        while sum(get_token_length(selected_sentences)) < token_limit:
            selected_sentences.append(
                sentences[labels == communities[com_idx]][label_index_lookup[com_idx]]
            )  ## Get the next sentence from the community

            selected_pmcids.append(
                pmcids[labels == communities[com_idx]][label_index_lookup[com_idx]]
            )

            label_index_lookup[com_idx] += 1
            com_idx += 1
            print(com_idx, len(communities))
            if com_idx >= len(communities):
                com_idx = 1

        ## pop the last one, since by definition we went over by including it
        selected_sentences.pop()

        return {
            "selected_sentences": selected_sentences,
            "selected_pmcids": selected_pmcids,
        }

    logging.info(f"{ent_id} has too many clusters to use round-robin selection")
    logging.info(
        f"Using greedy selection algorithm for {ent_id}. Only works on cluster centres."
    )

    ## If we're here, there are still too many tokens in the selection. Now we need to optimize for diversity and token count
    ## Use a greedy algorithm on the cluster centres, start with the first because it should be the largest cluster

    com_idx = 2  ## The -1 community is "noise", and we add those at index 1 manually so we start at 2
    selected_sentences = [sentences[labels == 0][0]]
    selected_pmcids = [pmcids[labels == 0][0]]
    selected_embeddings = [embeddings[labels == 0][0]]
    selected_idxs = [0]
    total_tokens = sum(get_token_length(selected_sentences))
    while total_tokens < token_limit:
        lengths = get_token_length(sentences[labels == communities[com_idx]])
        ## This loop runs for each community, and selects the most distinct sentence from that community
        for selected_embedding in selected_embeddings:
            distances = []
            cost = []
            idx_to_copy = []
            for idx, embedding in enumerate(embeddings[labels == communities[com_idx]]):
                if idx in selected_idxs:
                    continue
                distances.append(
                    util.pairwise_dot_score(selected_embedding, embedding)
                    .numpy()
                    .tolist()
                )
                cost.append(distances[-1] * lengths[idx])
                idx_to_copy.append(idx)
        next_selection = idx_to_copy[np.argmin(cost)]
        selected_sentences.append(
            sentences[labels == communities[com_idx]][next_selection]
        )
        selected_embeddings.append(
            embeddings[labels == communities[com_idx]][next_selection]
        )
        selected_pmcids.append(pmcids[labels == communities[com_idx]][next_selection])

        ## Check we aren't about to run out of tokens
        total_tokens = sum(get_token_length(selected_sentences))
        if total_tokens >= token_limit:
            selected_sentences.pop()
            break
        com_idx += 1
        ## If we've run out of clusters, go back to the first one
        if com_idx == len(communities) - 2:  ## -2 because we skip the -1 community
            com_idx = 1
    return {
        "selected_sentences": selected_sentences,
        "selected_pmcids": selected_pmcids,
    }
