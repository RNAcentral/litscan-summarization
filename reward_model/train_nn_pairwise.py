import pathlib

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)
from transformers.modeling_outputs import TokenClassifierOutput

from evaluate import load as load_metric


class PairwiseSummaryEvaluator(nn.Module):
    def __init__(self, base_model, num_classes=1, device="cpu", weights=None):
        super(PairwiseSummaryEvaluator, self).__init__()
        self.model = model = AutoModel.from_pretrained(
            base_model,
            config=AutoConfig.from_pretrained(
                base_model,
                output_attentions=True,
                output_hidden_states=True,
                device=device,
            ),
        )

        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(768, num_classes)
        self.num_classes = num_classes
        self.weights = weights

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
    ):
        ## Treat pretrained model as frozen
        with torch.no_grad():
            summ_output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        ## Get the second to last hidden layer (see https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning)
        summ_vec = summ_output[2][-2][:, 0, :]
        ## The last set of indices should be getting the embedding of the CLS token - i.e. sentence level?

        logits = self.classifier(self.dropout(summ_vec))

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )  # hidden_states=outputs.hidden_states,attentions=outputs.attentions)


def tokenize(x, tokenizer, max_seq_length=2048):
    tokenization = tokenizer(
        x["summary_1"] + "\n\n" + x["summary_2"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    x["input_ids"] = tokenization["input_ids"]
    x["attention_mask"] = tokenization["attention_mask"]
    x["labels"] = float(x["label"])  # - 1
    return x


def binarize_at_threshold(example, threshold=1):
    example["labels"] = 1 if example["labels"] >= threshold else 0
    return example


@click.command()
@click.option("--pretrained_model", default="allenai/longformer-base-4096")
@click.option("--max_seq_length", default=4098)
@click.option("--batch_size", default=4)
@click.option("--device", default="mps")
@click.option("--num_epochs", default=3)
@click.option("--base_lr", default=1e-5)
@click.option("--weight_decay", default=0.01)
@click.option("--output_dir", default="output", type=pathlib.Path)
def main(
    pretrained_model,
    max_seq_length,
    batch_size,
    device,
    num_epochs,
    base_lr,
    weight_decay,
    output_dir,
):
    config = AutoConfig.from_pretrained(pretrained_model)
    config.max_position_embeddings = max_seq_length
    config.torch_dtype = "fp16"
    summary_rating_data = load_dataset(
        "parquet",
        data_files={
            "train": "train_andrew_pairwise.pq",
            "test": "test_andrew_pairwise.pq",
        },
    )

    if pretrained_model == "mosaicml/mosaic-bert-base-seqlen-2048":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    summary_rating_data = summary_rating_data.map(
        lambda x: tokenize(x, tokenizer, max_seq_length)
    ).with_format("torch")

    d = pl.read_parquet("train_andrew_pairwise.pq")
    d = d.with_columns(
        tokens=pl.struct("summary_1", "summary_2").apply(
            lambda x: len(tokenizer.encode(x["summary_1"]))
            + len(tokenizer.encode(x["summary_2"]))
        )
    )

    print(d.describe())

    labels = np.array(summary_rating_data["train"]["labels"])
    # weights = np.bincount(labels-1) / len(labels)
    # print(1.0/weights)
    # plt.bar(np.arange(2), np.bincount(labels))
    # plt.show()
    # exit()

    print(config, pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        config=config,
    )

    summary_rating_data = summary_rating_data.remove_columns(
        ["feedback_1", "feedback_2", "summary_1", "summary_2", "label"]
    ).with_format("torch")
    model = PairwiseSummaryEvaluator(pretrained_model)

    total_params = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
    )  # sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")

    # exit()

    train_dataloader = DataLoader(
        summary_rating_data["train"], batch_size=batch_size, shuffle=True
    )
    eval_dataloader = DataLoader(
        summary_rating_data["test"], batch_size=batch_size, shuffle=False
    )

    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar_train = tqdm(range(num_training_steps))
    progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.03 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    metric_train = load_metric("mse")
    metric_eval = load_metric("mse")

    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []
    prev_max_acc = 0.0
    model.half()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            train_losses.append(loss.item())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric_train.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_train.update(1)
        train_accs.append(metric_train.compute()["mse"])
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            eval_losses.append(outputs.loss.item())
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric_eval.add_batch(predictions=predictions, references=batch["labels"])
            progress_bar_eval.update(1)

        eval_accs.append(metric_eval.compute()["mse"])
        max_acc = np.max(eval_accs)
        if max_acc > prev_max_acc:
            prev_max_acc = max_acc
            torch.save(model, output_dir / f"model_ep{epoch}_ac{max_acc}.pt")

    train_steps = np.arange(len(train_losses))
    eval_steps = np.arange(len(eval_losses)) * len(train_losses) / len(eval_losses)

    plt.plot(train_steps, train_losses, label="Train")
    plt.plot(eval_steps, eval_losses, label="Eval")

    plt.xlabel("Training steps")
    plt.ylabel("Loss (Crossentropy)")
    plt.legend()

    ## Save the model and other outputs
    output_dir.mkdir(exist_ok=True)
    torch.save(model, output_dir / "model.pt")
    loss_curve_location = output_dir / "losscurves.png"
    plt.savefig(loss_curve_location)

    plt.figure()
    plt.plot(np.arange(num_epochs), train_accs, label="Train")
    plt.plot(np.arange(num_epochs), eval_accs, label="Eval")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(output_dir / "accuracycurves.png")

    ## Build and save dataframe with training and eval losses
    loss_df = pl.DataFrame({"train_loss": train_losses})
    eval_loss_df = pl.DataFrame({"eval_loss": eval_losses})
    loss_df.write_parquet(output_dir / "train_losses.pq")
    eval_loss_df.write_parquet(output_dir / "eval_losses.pq")

    accu_df = pl.DataFrame({"train_acc": train_accs, "eval_acc": eval_accs})
    accu_df.write_parquet(output_dir / "accuracy.pq")


if __name__ == "__main__":
    main()
