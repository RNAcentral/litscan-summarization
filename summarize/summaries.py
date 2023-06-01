import logging
import re

from chains.summarization import (
    get_reference_chain,
    get_summarizer_chain,
    get_veracity_chain,
    get_veracity_revision_chain,
)
from langchain.callbacks import get_openai_callback
from llm_abstraction.models import get_model
from utils.validation import validate_summary


def generate_summary(
    model_name,
    ent_id,
    context,
    evaluate_truth=False,
    max_rescue_attempts=4,
    extra_args={},
):
    """
    Runs the LLM chains to produce a summary, first the summarizer chain,
    then runs some checking for the correctness of references. If needed,
    it will then run the reference addition chain a few times until the references
    are adequately inserted.

    Optionally, we can run a checking chain to see if the summary makes factual
    statements supported by the context
    """
    summary_chain = get_summarizer_chain(
        get_model(
            model_name,
            {"temperature": 0.1, "presence_penalty": -2, "frequency_penalty": 1}
            | extra_args,
        ),
        verbose=True,
    )
    reference_chain = get_reference_chain(
        get_model(
            model_name,
            {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
            | extra_args,
        ),
        verbose=True,
    )
    veracity_chain = get_veracity_chain(
        get_model(
            model_name,
            {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
            | extra_args,
        ),
        verbose=True,
    )

    veracity_revision_chain = get_veracity_revision_chain(
        get_model(
            model_name,
            {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
            | extra_args,
        ),
        verbose=True,
    )
    total_tokens = 0
    cost = 0
    with get_openai_callback() as cb:
        summary = summary_chain.run(ent_id=ent_id, context_str=context)
        print(cb)
        total_tokens += cb.total_tokens
        cost += cb.total_cost

    validation = validate_summary(summary, context)
    attempt = 1
    while not all(validation.values()):
        if attempt >= max_rescue_attempts:
            logging.warning(
                f"Unable to generate a good summary for {ent_id}. Returning what we have and giving up"
            )
            # return summary
            break
        logging.warning(
            "Summary auto validation failed! Running reference insertion chain to rescue..."
        )
        with get_openai_callback() as cb:
            summary = reference_chain.run(
                ent_id=ent_id, context_str=context, summary=summary
            )
            print(cb)
            total_tokens += cb.total_tokens
            cost += cb.total_cost

        validation = validate_summary(summary, context)
        attempt += 1

    if evaluate_truth:
        ## Check to see if the summary makes factual sense
        ## First transform the summary into a bulleted list
        bullet_summary = "- " + summary.replace(". ", "\n- ")
        logging.info("Evaluating truthfulness of summary")
        with get_openai_callback() as cb:
            veracity_check_result = veracity_chain.run(
                ent_id=ent_id, bullet_summary=bullet_summary, original_context=context
            )
            print(cb)
            total_tokens += cb.total_tokens
            cost += cb.total_cost

        print(veracity_check_result)
        if re.search(r".*False.*", veracity_check_result):
            logging.warning("Untrue statements found in summary, revising accordingly")
            with get_openai_callback() as cb:
                summary = veracity_revision_chain.run(
                    checked_assertions=veracity_check_result, summary=summary
                )
                print(cb)
                total_tokens += cb.total_tokens
                cost += cb.total_cost

    return summary, cost, total_tokens
