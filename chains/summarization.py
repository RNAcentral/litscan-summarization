from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

system_instruction = ("You are an experienced academic and always provide references for each sentence you write. "
                        "You are a researcher who always answers in a factual and unbiased way. "
                        "Provide at least one reference per sentence you produce."
                    )

def get_summarizer_prompt() -> ChatPromptTemplate:
    context_padding = ("As an experienced academic who ALWAYS provides references for each sentence you write, "
                    "produce a summary from the text below, focusing on {rna_id} and using the references for each sentence. "
                    "The reference for each sentence in the text is given at the end of the sentence, enclosed by []. "
                    "You MUST provide at least one reference per sentence you produce. "
                    "Use only the information in the context given below. "
                    "Use 200 words or less.\n\n{context_str}\nSummary:"
                        )
    
    system_prompt = SystemMessagePromptTemplate.from_template(system_instruction)
    human_prompt = HumanMessagePromptTemplate.from_template(context_padding)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])


    return chat_prompt


def get_revision_prompt() -> ChatPromptTemplate:    
    revision_context = ("Given the summary, and its original context, rewrite the summary to include references at the end of each sentence. "
                    "References are provided in the original context, enclosed in []."
                    "Summary: {summary}\n"
                    "Original context:\n{context_str}\n"
                    "Revised Summary: "
                    )
    
    system_prompt = SystemMessagePromptTemplate.from_template(system_instruction)
    human_prompt = HumanMessagePromptTemplate.from_template(revision_context)

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

def get_checker_chain(llm):
    pass