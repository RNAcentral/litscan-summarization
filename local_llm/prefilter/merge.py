import click
import polars as pl


@click.command()
@click.option("--output", "-o", default="merged_classifications.pq")
@click.argument("files", nargs=-1)
def main(files, output):
    data_chunks = pl.scan_parquet(files)
    data_chunks.sink_parquet(output)


if __name__ == "__main__":
    main()
