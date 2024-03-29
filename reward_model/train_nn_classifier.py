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


class SiameseSummaryEvaluator(nn.Module):
    def __init__(self, base_model, num_classes=5):
        super(SiameseSummaryEvaluator, self).__init__()
        self.model = model = AutoModel.from_pretrained(
            base_model,
            config=AutoConfig.from_pretrained(
                base_model,
                output_attentions=True,
                output_hidden_states=True,
                device="mps",
            ),
        )

        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(2 * 768, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        summ_input_ids,
        summ_attention_mask,
        ctx_input_ids,
        ctx_attention_mask,
        labels=None,
    ):
        ## Treat pretrained model as frozen
        with torch.no_grad():
            summ_output = self.model(
                input_ids=summ_input_ids, attention_mask=summ_attention_mask
            )
            ctx_output = self.model(
                input_ids=ctx_input_ids, attention_mask=ctx_attention_mask
            )

        ## Get the second to last hidden layer (see https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning)
        summ_vec = summ_output[2][-2][:, 0, :]
        ctx_vec = ctx_output[2][-2][:, 0, :]
        ## The last set of indices should be getting the embedding of the CLS token - i.e. sentence level?

        logits = self.classifier(self.dropout(torch.cat((summ_vec, ctx_vec), dim=1)))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )  # hidden_states=outputs.hidden_states,attentions=outputs.attentions)


class SummaryEvaluator(nn.Module):
    def __init__(self, base_model, num_classes=5, device="cpu", weights=None):
        super(SummaryEvaluator, self).__init__()
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
        summ_input_ids,
        summ_attention_mask,
        labels=None,
    ):
        ## Treat pretrained model as frozen
        with torch.no_grad():
            summ_output = self.model(
                input_ids=summ_input_ids, attention_mask=summ_attention_mask
            )

        ## Get the second to last hidden layer (see https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning)
        summ_vec = summ_output[2][-2][:, 0, :]
        ## The last set of indices should be getting the embedding of the CLS token - i.e. sentence level?

        logits = self.classifier(self.dropout(summ_vec))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.weights)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )  # hidden_states=outputs.hidden_states,attentions=outputs.attentions)


def tokenize(x, tokenizer, max_seq_length=4096):
    summ_tokenization = tokenizer(
        x["summary"], truncation=True, padding="max_length", max_length=max_seq_length
    )
    ctx_tokenization = tokenizer(
        x["context"], truncation=True, padding="max_length", max_length=max_seq_length
    )
    x["summ_input_ids"] = summ_tokenization["input_ids"]
    x["summ_attention_mask"] = summ_tokenization["attention_mask"]
    x["ctx_input_ids"] = ctx_tokenization["input_ids"]
    x["ctx_attention_mask"] = ctx_tokenization["attention_mask"]
    x["labels"] = x["feedback"]  # - 1
    return x


def concat_tokenize(x, tokenizer, max_seq_length=4096):
    tokenization = tokenizer(
        x["context"] + "\n\n" + x["summary"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    x["input_ids"] = tokenization["input_ids"]
    x["attention_mask"] = tokenization["attention_mask"]
    x["labels"] = x["feedback"]  # - 1
    return x


def binarize_at_threshold(example, threshold=1):
    example["labels"] = 1 if example["labels"] >= threshold else 0
    return example


@click.command()
@click.option("--pretrained_model", default="allenai/longformer-base-4096")
@click.option("--max_seq_length", default=4096)
@click.option("--batch_size", default=4)
@click.option("--device", default="mps")
@click.option("--num_epochs", default=3)
@click.option("--base_lr", default=1e-5)
@click.option("--weight_decay", default=0.01)
@click.option("--output_dir", default="output", type=pathlib.Path)
@click.option("--siamese_model", default=False, is_flag=True)
@click.option("--binarize", default=False, is_flag=True)
@click.option("--concat_model", default=False, is_flag=True)
def main(
    pretrained_model,
    max_seq_length,
    batch_size,
    device,
    num_epochs,
    base_lr,
    weight_decay,
    output_dir,
    siamese_model,
    binarize,
    concat_model,
):
    config = AutoConfig.from_pretrained(pretrained_model)
    config.max_position_embeddings = max_seq_length
    config.torch_dtype = "fp16"
    summary_rating_data = load_dataset(
        "parquet", data_files={"train": "train_andrew.pq", "test": "test_andrew.pq"}
    )

    if pretrained_model == "mosaicml/mosaic-bert-base-seqlen-2048":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if concat_model:
        summary_rating_data = summary_rating_data.map(
            lambda x: concat_tokenize(x, tokenizer, max_seq_length)
        )
    else:
        summary_rating_data = summary_rating_data.map(
            lambda x: tokenize(x, tokenizer, max_seq_length)
        )
    if binarize:
        summary_rating_data = summary_rating_data.map(binarize_at_threshold)

    labels = np.array(summary_rating_data["train"]["labels"])
    weights = np.bincount(labels - 1) / len(labels)
    print(1.0 / weights)
    # plt.bar(np.arange(2), np.bincount(labels))
    # plt.show()
    # exit()

    if siamese_model:
        summary_rating_data = summary_rating_data.remove_columns(
            ["summary", "context", "summary_id", "feedback"]
        ).with_format("torch")

        model = SiameseSummaryEvaluator(pretrained_model)
    elif concat_model:
        print(config)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            config=config,
        )
        summary_rating_data = summary_rating_data.remove_columns(
            ["summary", "context", "summary_id", "feedback"]
        ).with_format("torch")
    else:
        summary_rating_data = summary_rating_data.remove_columns(
            [
                "summary",
                "context",
                "summary_id",
                "feedback",
                "ctx_input_ids",
                "ctx_attention_mask",
            ]
        ).with_format("torch")
        if binarize:
            model = SummaryEvaluator(pretrained_model, num_classes=2)
        else:
            model = SummaryEvaluator(pretrained_model)

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

    metric_train = load_metric("accuracy", average="weighted")
    metric_eval = load_metric("accuracy", average="weighted")

    train_losses = []
    eval_losses = []
    train_accs = []
    eval_accs = []
    prev_max_acc = 0.0
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
        train_accs.append(metric_train.compute()["accuracy"])
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

        eval_accs.append(metric_eval.compute()["accuracy"])
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
