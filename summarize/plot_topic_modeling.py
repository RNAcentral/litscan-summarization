import click
import polars as pl
import umap
from sentence_selection.topic_modelling import run_topic_modelling
from sentence_selection.utils import plot
from sentence_transformers import SentenceTransformer


@click.command()
@click.option("--ent_id", multiple=True)
def main(ent_id):
    print(ent_id)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
    df = (
        pl.read_json("raw_sentences.json")
        .filter(pl.col("primary_id").is_in(ent_id))
        .explode("sentence")
    )

    result, exemplars = run_topic_modelling(df, model)

    embeddings = model.encode(
        result.get_column("sentence").to_list(), show_progress_bar=True
    )

    umap_display = umap.UMAP(
        n_neighbors=20, n_components=2, min_dist=0.0, metric="cosine"
    ).fit_transform(embeddings)

    result = result.with_columns(
        x=pl.Series(umap_display[:, 0]), y=pl.Series(umap_display[:, 1])
    )

    outliers = result.filter(pl.col("sentence_labels") == -1)
    clustered = result.filter(pl.col("sentence_labels") != -1)

    plot(clustered, outliers)


if __name__ == "__main__":
    main()
