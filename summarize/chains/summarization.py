from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_instruction = (
    "You are an experienced academic and always provide references for each sentence you write. "
    "You are a researcher who always answers in a factual and unbiased way. "
    "Provide at least one reference per sentence you produce."
)
## 39 tokens


def get_summarizer_prompt() -> ChatPromptTemplate:
    context_padding = (
        "As an experienced academic who ALWAYS provides references for each sentence you write, "
        "produce a summary from the text below, focusing on {ent_id} and using the references for each sentence. "
        "The reference for each sentence in the text is given at the end of the sentence, enclosed by []. "
        "You MUST provide at least one reference per sentence you produce. "
        "Use only the information in the context given below. "
        "Use 200 words or less.\n\n{context_str}\n"
        "\nSummary:"
    )
    # 91 tokens - call it 100 for ID as well

    system_prompt = SystemMessagePromptTemplate.from_template(system_instruction)
    human_prompt = HumanMessagePromptTemplate.from_template(context_padding)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    return chat_prompt


def get_revision_prompt() -> ChatPromptTemplate:
    revision_context = (
        "Given the following summary:\n{summary}\n"
        "and its original context: \n{context_str}\n"
        "rewrite the summary to include at least one reference at the end of each sentence. "
        "References are provided in the original context, enclosed in [].\n"
        "Revised Summary: "
    )

    system_prompt = SystemMessagePromptTemplate.from_template(system_instruction)
    human_prompt = HumanMessagePromptTemplate.from_template(revision_context)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    return chat_prompt


def get_veracity_prompt() -> ChatPromptTemplate:
    """
    This gives us the prompt template for checking the veracity of the generated summary,
    given the context from which it was created.
    """
    system_instruction_veracity = (
        "You are an experienced academic who has been asked to fact check a summary. "
        "You will check the validity of claims made, and that the claims have appropriate references. "
        "When making your assertions, you will only use the provided context, and will not use external sources"
    )
    veracity_context = (
        "Here is a bullet point list of statements about the entity {ent_id}:\n"
        "{bullet_summary}\n\n"
        "The summary was derived from the following context:\n"
        "{original_context}\n"
        "For each statement, determine whether it is true or false, based on whether there is supporting evidence in the context. "
        "Make a determination for all statements, If a statement is false, explain why.\n\n"
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        system_instruction_veracity
    )

    human_prompt = HumanMessagePromptTemplate.from_template(veracity_context)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    return chat_prompt


def get_veracity_revision_prompt() -> ChatPromptTemplate:
    """
    Uses the summary and the veracity check output to refine the summary. This shouldn't need the context I don't think, since the reasoning
    for claims made in the veracity check should be in the returned veracity statement.

    """
    system_prompt = SystemMessagePromptTemplate.from_template(system_instruction)
    veracity_revision_context = (
        "{checked_assertions}\n\n"
        "In light of the above checks about its veracity, refine the summary below to ensure all statements are true.\n"
        "Original summary: \n{summary}\n"
        "Revised summary:\n"
    )
    human_prompt = HumanMessagePromptTemplate.from_template(veracity_revision_context)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    return chat_prompt


def get_summarizer_chain(llm, verbose=False) -> LLMChain:
    prompt = get_summarizer_prompt()
    print(llm)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return chain


def get_reference_chain(llm, verbose=False) -> LLMChain:
    prompt = get_revision_prompt()
    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return chain


def get_veracity_chain(llm, verbose=False) -> LLMChain:
    prompt = get_veracity_prompt()
    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return chain


def get_veracity_revision_chain(llm, verbose=False) -> LLMChain:
    prompt = get_veracity_revision_prompt()
    chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return chain