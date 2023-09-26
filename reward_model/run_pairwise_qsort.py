"""
Load the model as a comparison function, use it as the basis of
a quicksort algorithm.
"""

import pathlib
from functools import cmp_to_key

import click
import polars as pl
import torch
from datasets import load_dataset
from train_nn_pairwise import PairwiseSummaryEvaluator
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)


class SummaryCollection(object):
    def __init__(self, dataframe):
        """
        Copy eveyrthing out of a dataframe
        """
        self.ent_id = dataframe["ent_id"].to_list()
        self.summaries = dataframe["summary"].to_list()
        self.urs_taxid = dataframe["urs_taxid"].to_list()

    def sort(self, comparison_function):
        """
        Sort summaries using the comparison function and make the same rearrangement in the other lists
        """
        self.sorted_summaries = quicksort(self.summaries, comparison_function)
        indices = [self.summaries.index(x) for x in self.sorted_summaries]
        self.sorted_ent_id = [self.ent_id[i] for i in indices]
        self.sorted_urs_taxid = [self.urs_taxid[i] for i in indices]
        self.summaries = self.sorted_summaries

    def to_dataframe(self):
        return pl.DataFrame(
            {
                "ent_id": self.sorted_ent_id,
                "summary": self.summaries,
                "urs_taxid": self.sorted_urs_taxid,
            }
        )


def tokenize(s1, s2, tokenizer, max_seq_length=2048):
    x = {}
    tokenization = tokenizer(
        s1 + "\n\n" + s2,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    x["input_ids"] = tokenization["input_ids"]
    x["attention_mask"] = tokenization["attention_mask"]
    return x


def load_model(model_path, device="cpu"):
    model = torch.load(model_path, map_location=torch.device(device))
    model = model.to(device)
    model.eval()
    return model


def compare_summaries(model, tokenizer, summary_1, summary_2, device="cpu"):
    """
    Compare two summaries, return 1 if summary_1 is better, -1 if summary_2 is better,
    and 0 if they are equal.
    """
    tokenization = tokenize(summary_1, summary_2, tokenizer)
    input_ids = torch.tensor(tokenization["input_ids"]).unsqueeze(0).to(device)
    attention_mask = (
        torch.tensor(tokenization["attention_mask"]).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        output = (
            torch.argmax(
                torch.nn.functional.softmax(model(input_ids, attention_mask).logits)
            )
            .cpu()
            .numpy()
        )

    if output == 0:
        return 1
    elif output == 1:
        return -1
    else:
        return 0


@click.command()
@click.option("--summary_data", default="summaries.pq", type=pathlib.Path)
@click.option("--model_path", default=".", type=pathlib.Path)
@click.option("--pretrained_model", default="allenai/longformer-base-4096")
@click.option("--device", default="mps")
@click.option("--output", default="output", type=pathlib.Path)
def main(
    summary_data,
    model_path,
    pretrained_model,
    device,
    output,
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = load_model(model_path, device)
    data = pl.read_parquet(summary_data)
    print(data)
    data = data.filter(pl.col("user_id").eq("Nancy") & pl.col("summary_id").gt(300))
    data = data.select(
        [
            "summary",
            "feedback",
            "summary_id",
        ]
    ).unique()

    ## Convert to summary collection, use sort function, then convert back
    summary_collection = SummaryCollection(data)
    summary_collection.sort(
        lambda x, y: compare_summaries(model, tokenizer, x, y, device)
    )
    data = summary_collection.to_dataframe()
    data.write_parquet(output)


def quicksort(a_list, comparison_function):
    """
    Sort a list using a comparison function.
    """
    if len(a_list) <= 1:
        return a_list
    pivot = a_list[0]
    left = [x for x in a_list[1:] if comparison_function(x, pivot) == -1]
    right = [x for x in a_list[1:] if comparison_function(x, pivot) != -1]
    return (
        quicksort(left, comparison_function)
        + [pivot]
        + quicksort(right, comparison_function)
    )


if __name__ == "__main__":
    main()
    # import numpy as np
    # l = np.random.randint(0, 100, 100)
    # print(l)
    # print(quicksort(l, lambda x,y: 1 if x > y else -1))
