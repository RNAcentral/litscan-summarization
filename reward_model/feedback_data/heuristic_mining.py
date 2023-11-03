import re
from enum import Enum

import matplotlib.pyplot as plt
import polars as pl
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from sklearn.model_selection import train_test_split
from snorkel.labeling import LFAnalysis, labeling_function

sentence_model = SentenceTransformer("paraphrase-MiniLM-L12-v2", device="mps")


ABSTAIN = -1
TERRIBLE = 0
BAD = 1
OK = 2
GOOD = 3
EXCELLENT = 4

"""
Utility function to get sources from a summary, just grabs everything in square brackets and returns them as a set
"""


def get_references(text):
    references = re.findall(r"\[.*?\]", text)
    return set(references)


def find_redundant_sentences(summary):
    """
    Split into sentences, then encode each into a vector and see if any have very high similarity
    If they have high semantic similarity, they are probably redundant
    """
    sentences = summary.split(".")
    sentence_vectors = sentence_model.encode(sentences)
    redundant_sentences = []

    for i, sentence in enumerate(sentences):
        for j, other_sentence in enumerate(sentences):
            if i != j and j > i:
                similarity = util.cos_sim([sentence_vectors[i]], [sentence_vectors[j]])[
                    0
                ][0]
                if similarity > 0.85:
                    redundant_sentences.append((sentence, other_sentence, similarity))

    return len(redundant_sentences) / len(sentences)


@labeling_function()
def summary_sentence_support_in_context(x):
    """
    Split into sentences, then encode each into a vector and see if any have very high similarity
    If they have high semantic similarity, they are probably redundant
    """
    summary, context = x.summary, x.context
    summary_sentences = summary.split(".")
    summary_sentence_vectors = sentence_model.encode(summary_sentences)

    context_sentences = context.split(".")
    context_sentence_vectors = sentence_model.encode(context_sentences)

    supports = []
    for i, ssv in enumerate(summary_sentence_vectors):
        s1_support = []
        for j, csv in enumerate(context_sentence_vectors):
            similarity = util.cos_sim([ssv], [csv])[0][0]
            if similarity > 0.55:
                s1_support.append(True)
            else:
                s1_support.append(False)
        supports.append(s1_support)

    if all([any(s) for s in supports]):
        return ABSTAIN
    else:
        return TERRIBLE


@labeling_function()
def talks_about_us(x):
    """Returns True if the text contains a phrase that talks about the us."""
    if " us " in x.summary.lower() or " we " in x.summary.lower():
        return BAD
    else:
        return ABSTAIN


@labeling_function()
def talks_about_present_study(x):
    """If the test contains the words 'the present study' or 'this study', it is bad."""
    if (
        "study" in x.summary.lower()
        or "assayed" in x.summary.lower()
        or "future study" in x.summary.lower()
    ):
        return OK
    else:
        return ABSTAIN


@labeling_function()
def contains_fake_references(x):
    """
    References not present in the context but are present in the summary.
    Find everything contained in square brackets, make sure they match in the context.
    """
    summary_references = get_references(x.summary)
    context_references = get_references(x.context)

    if summary_references.issubset(context_references):
        return ABSTAIN
    else:
        return TERRIBLE


@labeling_function()
def uses_wrong_reference_style(x):
    """
    Find everything contained in square brackets and make sure they match the regex
    """
    summary_references = get_references(x.summary)

    for reference in summary_references:
        if not re.match(r"\[PMC\d+\]", reference):
            if "et al" in reference:
                return TERRIBLE
            else:
                return BAD
    return ABSTAIN


@labeling_function()
def diversity_of_sources(x):
    """
    When the context has many sources, but the summary doesn't, it should score poorly
    This one must be on a sliding scale, the more it misses the worse it is.
    """
    summary_references = get_references(x.summary)
    context_references = get_references(x.context)
    if len(summary_references) / len(context_references) < 0.1:
        return TERRIBLE
    elif len(summary_references) / len(context_references) < 0.3:
        return BAD
    else:
        return ABSTAIN


@labeling_function()
def irrelevant_statements(x):
    """
    When the summary contains things like require further investigation, methods stuff, selected for further analysis, etc.
    It can get at most a 4 (Good)
    """
    if (
        "further investigation" in x.summary.lower()
        or "further analysis" in x.summary.lower()
        or "were measured" in x.summary.lower()
        or "qpcr" in x.summary.lower()
    ):
        return OK
    else:
        return ABSTAIN


@labeling_function()
def contains_hyperlinks(x):
    """Returns True if the text contains a hyperlink."""
    if "http" in x.summary.lower() or "www" in x.summary.lower():
        return BAD
    else:
        return ABSTAIN


@labeling_function()
def restates_references(x):
    """If the summary has References:\n it scores OK."""
    if "References:" in x.summary:
        return OK
    else:
        return ABSTAIN


@labeling_function()
def has_redundant_info(x):
    """
    Use sentence similarity to look for redundant information in the summary
    """
    redundant_sentences = find_redundant_sentences(x.summary)
    if redundant_sentences > 0.3:
        return BAD
    else:
        return ABSTAIN


### Things which make a summary good:
@labeling_function()
def has_good_intro(x):
    """
    Summaries that start with XXX is a lncRNA (for example) are excellent
    """
    first_sentence = x.summary.split(".")[0]
    if "is a" in first_sentence:
        return EXCELLENT
    else:
        return ABSTAIN


@labeling_function()
def synthesised_references(x):
    """
    Look for summaries which have multiple references at the end of sentences, usually [][] or with [PMC...,PMC...]
    """
    multi_refs = re.findall(r"\[.*?\]\[.*?\]", x.summary)
    if len(multi_refs) > 0:
        return EXCELLENT
    else:
        return ABSTAIN


@labeling_function()
def has_wrap_up(x):
    """
    Look for summaries which have a wrap up sentence, usually starting with 'In conclusion'
    """
    if "In summary" in x.summary.split(".")[-2] or x.summary.split(".")[-2].startswith(
        "Overall"
    ):
        return EXCELLENT
    else:
        return ABSTAIN


from snorkel.labeling import PandasLFApplier

lfs = [
    # restates_references,
    #    contains_hyperlinks,
    irrelevant_statements,
    #    talks_about_us,
    talks_about_present_study,
    #     diversity_of_sources,
    # #    contains_fake_references,
    #    uses_wrong_reference_style,
    has_good_intro,
    synthesised_references,
    has_wrap_up,
    has_redundant_info,
    summary_sentence_support_in_context,
]

df = pl.read_csv("all_feedback_100.csv").with_columns(
    pl.col("user_id").apply(lambda x: "Nancy" if x == "anonymous" else x)
)
df = df.filter(pl.col("user_id") == "andrew").unique("rna_id").sort("feedback_id")

print(
    df.filter(pl.col("free_feedback").str.contains("wrong"))
    .get_column("feedback_id")
    .to_list()
)
print(df.filter(pl.col("free_feedback").str.contains("repetitive")).height / df.height)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=123)
print(
    df_train.filter(pl.col("free_feedback").str.contains("wrong")).height
    / df_train.height
)

# exit()

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(
    df=df_train.to_pandas(),
)

print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())


# plt.imshow(L_train, cmap="Blues", aspect="auto")
# plt.show()


L_test = applier.apply(df=df_test.to_pandas())
Y_test = df_test.get_column("feedback").to_numpy() - 1


# coverage_restate, coverage_hyperlinks, coverage_irrelevant = (L_train != ABSTAIN).mean(axis=0)
# print(f"restates_references coverage: {coverage_restate * 100:.1f}%")
# print(f"contains_hyperlinks coverage: {coverage_hyperlinks * 100:.1f}%")
# print(f"irrelevant_statements coverage: {coverage_irrelevant * 100:.1f}%")

from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter(cardinality=5)
preds_train, probs_train = majority_model.predict(L=L_train, return_probs=True)

majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="abstain")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=5, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)


label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="abstain")[
    "accuracy"
]

preds_test = label_model.predict(L=L_test)


print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")


from snorkel.labeling import filter_unlabeled_dataframe

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train.to_pandas(), y=probs_train, L=L_train
)

print(df_train_filtered.shape)

print(preds_test, Y_test)
