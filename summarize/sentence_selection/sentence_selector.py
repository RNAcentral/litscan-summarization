"""
Uses output of topic modelling to select sentences for each ID

Input: JSON file containing sentences grouped by ID, and topic modelled
Output: JSON file containing sentences selected for each ID

"""

import logging

logging.getLogger().setLevel(logging.INFO)
import numpy as np
import polars as pl
from sentence_selection.topic_modelling import run_topic_modelling
from sentence_selection.utils import get_token_length
from sentence_transformers import util


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
    ent_id = row["primary_id"]
    del row["primary_id"]
    row = pl.DataFrame(row)
    sentences = row.get_column("sentence").to_list()
    pmcids = row.get_column("pmcid").to_list()

    ## If we don't have enough, return all
    if sum(get_token_length(sentences)) <= token_limit:
        logging.info(f"Few tokens for {ent_id}, using all sentences")
        return {
            "selected_sentences": sentences,
            "selected_pmcids": pmcids,
            "method": "all",
        }

    ## If we have too many, use topic modelling
    logging.info(f"Too many tokens for {ent_id}, using topic modelling")
    row, communities = run_topic_modelling(row, model)
    ## Catch the case where there are nil communities
    if len(communities) == 0:
        logging.info(
            f"No communities for {ent_id}, re-running with smaller minimum cluster size"
        )
        row, communities = run_topic_modelling(
            row, model, min_cluster_size=3, min_samples=1
        )
    sentences = np.array(sentences)
    pmcids = np.array(pmcids)

    ## Try to reduce the number of sentences that need to be encoded by clustering and taking exemplars
    ## Exemplar indices are in the communities list - one list of exemplars for each cluster

    embeddings = row.get_column("embeddings").to_numpy()

    ## Select a starting sentence per cluster
    selected_sentences = [
        sentences[c[0]] for c in communities
    ]  ## See if the noise community is in there...
    selected_pmcids = [pmcids[c[0]] for c in communities]
    print(sum(get_token_length(selected_sentences)))
    ## If there aren't too many clusters, this will be true, and we can select from them until we run out of tokens
    if sum(get_token_length(selected_sentences)) <= token_limit:
        logging.info(f"Sampling communities for {ent_id}, until token limit")
        ## First, pop all the first sentences 'cause they're already in the list
        [c.pop(0) for c in communities]

        ## round-robin grabbing of sentences until we hit the limit
        while sum(get_token_length(selected_sentences)) < token_limit:
            for c in communities:
                if len(c) > 0:  ## If there are sentences left in the community
                    idx = c.pop(0)
                    selected_sentences.append(sentences[idx])
                    selected_pmcids.append(pmcids[idx])
            if all([len(c) == 0 for c in communities]):
                break  ## This would mean taking all the exemplars still doesn't hit the token limit

        if sum(get_token_length(selected_sentences)) > token_limit:
            logging.info(f"Too many sentences for {ent_id}, removing last sentence")
            ## pop the last one, since by definition we went over by including it
            selected_sentences.pop()
            selected_pmcids.pop()

        return {
            "selected_sentences": selected_sentences,
            "selected_pmcids": selected_pmcids,
            "method": "round-robin",
        }

    logging.info(f"{ent_id} has too many clusters to use round-robin selection")
    logging.info(
        f"Using greedy selection algorithm for {ent_id}. Only works on cluster centres."
    )

    ## If we're here, there are still too many tokens in the selection. Now we need to optimize for diversity and token count
    ## Use a greedy algorithm on the cluster centres, start with the first because it should be the largest cluster

    start_idx = communities[0].pop()  ## This will always be selected, so ok to pop
    selected_sentences = [sentences[start_idx]]
    selected_pmcids = [pmcids[start_idx]]
    selected_embeddings = [embeddings[start_idx]]
    selected_idxs = [start_idx]

    total_tokens = sum(get_token_length(selected_sentences))
    while total_tokens < token_limit:
        lengths = get_token_length(selected_sentences)
        ## This loop runs for each community, and selects the most distinct sentence from that community
        for e_idx, selected_embedding in enumerate(selected_embeddings):
            distances = []
            cost = []
            community_idx = []
            idx_to_copy = []
            for c_idx, comm in enumerate(communities):
                if (
                    len(comm) == 0
                ):  ## If we've run out of sentences in the community, skip it
                    continue
                idx = comm[0]  ## Get the first sentence in the community
                if (
                    idx in selected_idxs
                ):  ## This index should be to the sentences, so this ought to be impossible...
                    continue
                embedding = embeddings[idx]
                distances.append(
                    util.pairwise_dot_score(selected_embedding, embedding)
                    .numpy()
                    .tolist()
                )
                cost.append(distances[-1] * lengths[e_idx])
                idx_to_copy.append(idx)
                community_idx.append(c_idx)
        ## Get the index of the minimum, use to grab the right sentence and community
        min_index = np.argmin(cost)
        comm_index = community_idx[min_index]
        communities[comm_index].pop(
            0
        )  ## Remove the selected sentence from the community
        next_selection = idx_to_copy[min_index]

        ## Put the selection in...
        selected_sentences.append(sentences[next_selection])
        selected_embeddings.append(embeddings[next_selection])
        selected_pmcids.append(pmcids[next_selection])
        selected_idxs.append(next_selection)

        ## Check we aren't about to run out of tokens
        total_tokens = sum(get_token_length(selected_sentences))
        if total_tokens >= token_limit:
            selected_sentences.pop()
            selected_pmcids.pop()
            selected_idxs.pop()
            selected_embeddings.pop()
            break
    return {
        "selected_sentences": selected_sentences,
        "selected_pmcids": selected_pmcids,
        "method": "greedy:",
    }
