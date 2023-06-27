import click
import hdbscan
import numpy as np
import polars as pl
import umap
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


## Some functions for doing TF-IDF
def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(
        documents
    )
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.select(pl.col("topic")).to_numpy().flatten())
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {
        label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1]
        for i, label in enumerate(labels)
    }
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (
        df.groupby(["topic"])
        .agg(pl.col("doc").count().alias("size"))
        .sort("size", descending=True)
    )
    return topic_sizes


def get_exemplars(cluster_object, embeddings):
    """
    Uses the exemplars from the HDBSCAN cluster object to get the indices of the sentences to use
    """
    exemplar_vectors = cluster_object.exemplars_
    exemplar_indices = []
    for cluster in exemplar_vectors:
        exemplar_indices.append([])
        for exemplar in cluster:
            exemplar_indices[-1].append(np.where(exemplar == embeddings)[0][0])
    return exemplar_indices


def cluster_sentences(id_sentences, model, min_cluster_size=15, min_samples=5):
    """
    Clusters the sentences for a given ID and labels them in the dataframe
    """
    if isinstance(id_sentences, pl.Series):
        id_sentences = id_sentences.to_list()

    embeddings = (
        model.encode(
            id_sentences,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        .cpu()
        .numpy()
    )
    k = min(15, len(id_sentences) // 4)
    N = 20

    umap_embeddings = umap.UMAP(
        n_neighbors=k, n_components=N, metric="cosine"
    ).fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit(umap_embeddings)
    topics = get_topics(id_sentences, cluster.labels_)
    exemplar_indices = get_exemplars(cluster, umap_embeddings)
    selection = {
        "sentence_labels": cluster.labels_.tolist(),
        "topics": topics,
        "umap_embeddings": umap_embeddings,
        "exemplar_indices": exemplar_indices,
    }
    return selection


def get_topics(sentences, labels):
    """
    Returns the topics for a given set of sentences and labels
    """
    if isinstance(sentences[0], list):
        sentences = [s[0] for s in sentences]
    docs_df = pl.DataFrame(
        {"doc": sentences, "topic": labels, "doc_id": range(len(sentences))}
    )
    docs_per_topic = docs_df.groupby("topic").agg(pl.col("doc").str.concat(" "))

    ## Do the topic modeling
    tf_idf, count = c_tf_idf(
        docs_per_topic.select("doc").to_numpy().flatten(), m=len(sentences)
    )
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=3)
    topic_summary = [",".join([a[0] for a in top_n_words[l][:3]]) for l in labels]
    return topic_summary


def run_topic_modelling(sentences, model, min_cluster_size=15, min_samples=5):
    cluster_res = cluster_sentences(
        sentences.get_column("sentence"), model, min_cluster_size, min_samples
    )
    exemplar_indices = cluster_res["exemplar_indices"]
    sentences = sentences.with_columns(
        sentence_labels=pl.Series(cluster_res["sentence_labels"]),
        topics=pl.Series(cluster_res["topics"]),
        embeddings=pl.Series(cluster_res["umap_embeddings"]),
    )
    return sentences, exemplar_indices


@click.command()
@click.option(
    "--json_path",
    type=click.Path(exists=True),
    help="Path to JSON file containing sentences. Should be pre-grouped by ID",
)
@click.option(
    "--device", type=str, default="cpu:0", help="Device to use for sentence transformer"
)
@click.option("--output_path", type=click.Path(), help="Path to output file")
def main(json_path, device, output_path):
    sentences = pl.read_json(json_path)
    print(sentences)
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    sentences = run_topic_modelling(sentences, model)
    sentences.write_json(output_path)


if __name__ == "__main__":
    main()
