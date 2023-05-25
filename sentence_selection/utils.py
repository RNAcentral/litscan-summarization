import polars as pl
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def get_token_length(sentences):
    return [len(enc.encode(s)) for s in sentences]


def prefilter_sentences(df, regexes=None):
    """
    Use a regex to remove sentences that are found in tables, figures or supplementary material

    Optionally apply some user supplied regexes to remove sentences that match those regexes.
    """
    filtered_df = df.lazy().filter(
        pl.col("sentence")
        .arr.eval(
            ~pl.element().str.contains("image|figure|table|supplementary material")
        )
        .collect()
    )

    if regexes:
        for regex in regexes:
            filtered_df = (
                filtered_df.lazy()
                .filter(pl.col("sentence").str.contains(regex).is_not())
                .collect()
            )

    return filtered_df
