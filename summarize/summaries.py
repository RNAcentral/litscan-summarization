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

pmcid_pattern = re.compile(r"PMC\d+")


def validation_revise_summary(
    summary: str,
    context: str,
    validation: dict,
    total_tokens,
    cost,
    model_name: str,
    ent_id: str,
    first_ref: str,
    extra_args={},
):
    if not validation["adequate"]:
        reference_chain = get_reference_chain(
            get_model(
                model_name,
                {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
                | extra_args,
            ),
            "adequate",
            verbose=True,
        )
    elif not validation["real"]:
        reference_chain = get_reference_chain(
            get_model(
                model_name,
                {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
                | extra_args,
            ),
            "fake",
            verbose=True,
        )
    else:
        reference_chain = get_reference_chain(
            get_model(
                model_name,
                {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
                | extra_args,
            ),
            "other",
            verbose=True,
        )
    with get_openai_callback() as cb:
        prompt = (
            reference_chain.prep_prompts(
                [
                    {
                        "ent_id": ent_id,
                        "context_str": context,
                        "summary": summary,
                        "first_ref": first_ref,
                    }
                ]
            )[0][0]
            .to_messages()[1]
            .content
        )
        summary = reference_chain.run(
            ent_id=ent_id, context_str=context, summary=summary, first_ref=first_ref
        )
        print(cb)
        total_tokens += cb.total_tokens
        cost += cb.total_cost

    return prompt, summary, total_tokens, cost


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
    first_ref = pmcid_pattern.findall(context)[0]
    with get_openai_callback() as cb:
        summary = summary_chain.run(
            ent_id=ent_id, context_str=context, first_ref=first_ref
        )
        print(cb)
        total_tokens += cb.total_tokens
        cost += cb.total_cost

    validation = validate_summary(summary, context)
    problem_summary = False
    attempt = 1
    print(validation)
    rescue_prompts = []
    while not all(validation.values()):
        if attempt >= max_rescue_attempts:
            logging.warning(
                f"Unable to generate a good summary for {ent_id}. Returning what we have and giving up"
            )
            # return summary
            problem_summary = True
            break
        logging.warning(
            "Summary auto validation failed! Running reference insertion chain to rescue..."
        )
        # if not validation["adequate"]:
        #     reference_chain = get_reference_chain(
        #         get_model(
        #             model_name,
        #             {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
        #             | extra_args,
        #         ),
        #         "adequate",
        #         verbose=True,
        #     )
        # elif not validation["real"]:
        #     reference_chain = get_reference_chain(
        #         get_model(
        #             model_name,
        #             {"temperature": 0.1, "presence_penalty": 0, "frequency_penalty": 0}
        #             | extra_args,
        #         ),
        #         "fake",
        #         verbose=True,
        #     )
        # with get_openai_callback() as cb:
        #     summary = reference_chain.run(
        #         ent_id=ent_id, context_str=context, summary=summary, first_ref=first_ref
        #     )
        #     print(cb)
        #     total_tokens += cb.total_tokens
        #     cost += cb.total_cost
        print(validation)
        prompt, summary, total_tokens, cost = validation_revise_summary(
            summary,
            context,
            validation,
            total_tokens,
            cost,
            model_name,
            ent_id,
            first_ref,
            extra_args,
        )
        rescue_prompts.append(prompt)
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
        truthful = not (
            re.search(r".*False.*", veracity_check_result)
            or re.search(r".*Misleading.*", veracity_check_result)
        )
        if not truthful:
            logging.warning(
                "Untrue/misleading statements found in summary, revising accordingly"
            )
            with get_openai_callback() as cb:
                summary = veracity_revision_chain.run(
                    checked_assertions=veracity_check_result,
                    summary=summary,
                    first_ref=first_ref,
                )
                print(cb)
                total_tokens += cb.total_tokens
                cost += cb.total_cost
            validation = validate_summary(summary, context)
            while not all(validation.values()):
                if attempt >= max_rescue_attempts:
                    logging.warning(
                        f"Unable to generate a good summary for {ent_id}. Returning what we have and giving up"
                    )
                    # return summary
                    problem_summary = True
                    break
                logging.warning(
                    "Summary auto validation failed! Running reference insertion chain to rescue..."
                )
                prompt, summary, total_tokens, cost = validation_revise_summary(
                    summary,
                    context,
                    validation,
                    total_tokens,
                    cost,
                    model_name,
                    ent_id,
                    first_ref,
                    extra_args,
                )
                rescue_prompts.append(prompt)

    return (
        summary,
        cost,
        total_tokens,
        attempt,
        rescue_prompts,
        problem_summary,
        truthful,
        veracity_check_result,
    )
