import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_scheduler
from transformers.modeling_outputs import TokenClassifierOutput

from evaluate import load as load_metric

_pretrained_model = "allenai/longformer-base-4096"
max_seq_length = 4096
batch_size = 4
device = "mps"


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
        self.mix = nn.Linear(2 * 768, 512)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_classes)
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
        ## The last set of indices should be getting the embedding of the CLS token

        ## Mix the vectors
        mixed = self.mix(torch.cat((summ_vec, ctx_vec), dim=1))
        # mixed = self.dropout(torch.max(mixed, dim=1)[0])

        logits = self.classifier(mixed)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
        )  # hidden_states=outputs.hidden_states,attentions=outputs.attentions)


def tokenize(x):
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
    x["labels"] = x["feedback"] - 1
    return x


config = AutoConfig.from_pretrained(_pretrained_model)

summary_rating_data = load_dataset(
    "parquet", data_files={"train": "train_andrew.pq", "test": "test_andrew.pq"}
)
tokenizer = AutoTokenizer.from_pretrained(_pretrained_model)

summary_rating_data = summary_rating_data.map(tokenize)
summary_rating_data = summary_rating_data.remove_columns(
    ["summary", "context", "summary_id", "feedback"]
).with_format("torch")


model = SiameseSummaryEvaluator(_pretrained_model)

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
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))


optimizer = AdamW(model.parameters(), lr=0.1)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

metric_train = load_metric("accuracy", average="weighted")
metric_eval = load_metric("accuracy", average="weighted")

train_losses = []
eval_losses = []

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
    print(metric_train.compute())
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

    print(metric_eval.compute())


train_steps = np.arange(len(train_losses))
eval_steps = np.arange(len(eval_losses))

plt.plot(train_steps, train_losses)
plt.plot(eval_steps, eval_losses)

plt.show()
