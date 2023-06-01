import re


def build_context(sentences, pmcids, joining_str=" "):
    """
    Join sentences from the articles with an extraction note, then put the whole thing together in a context string
    """
    cited_sentences = []
    for sentence, pmcid in zip(sentences, pmcids):
        sentence = re.sub(r"\[.+\]", "", sentence).strip().replace("\n", " ")
        cited_sentences.append(f"{sentence.strip('.')} [{pmcid}].")

    context = joining_str.join(cited_sentences)
    return context
