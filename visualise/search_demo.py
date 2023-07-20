import os
import re
from collections import namedtuple
from random import choices

import gradio as gr
import psycopg2 as pg

"""
Its easier to just copy the relevant bits here...
"""

pmcid_pattern = re.compile(r"PMC\d+")

system_instruction = (
    "You are an experienced academic and always provide references for each sentence you write. "
    "You are a researcher who always answers in a factual and unbiased way. "
    "Provide at least one reference per sentence you produce."
)

context_padding = (
    "As an experienced academic who ALWAYS provides references for each sentence you write, "
    "produce a summary from the text below, focusing on {ent_id} and using the references for each sentence. "
    "\n\n{context_str}\n"
    "The reference for each sentence in the text is given at the end of the sentence, enclosed by []. "
    "For example, the first sentence has the reference [{first_ref}]. "
    "You should use the same format for references in your summary. "
    "You MUST provide at least one reference per sentence you produce."
    "Use only the information in the context given above. "
    "Use 200 words or less."
    "\nSummary:"
)

revision_context = (
    "Given the following summary:\n{summary}\n"
    "and its original context: \n{context_str}\n"
    "rewrite the summary to include at least one reference at the end of each sentence. "
    "References are provided in the original context, enclosed in [].\n"
    "For example, the first sentence has the reference [{first_ref}]. "
    "You should use the same format for references in your summary. "
    "Revised Summary: "
)

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

veracity_revision_context = (
    "{checked_assertions}\n\n"
    "In light of the above checks about its veracity, refine the summary below to ensure all statements are true.\n"
    "Original summary: \n{summary}\n"
    "Revised summary:\n"
)


Settings = namedtuple(
    "Settings",
    [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DATABASE",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "ENVIRONMENT",
    ],
)


def get_postgres_credentials(ENVIRONMENT):
    ENVIRONMENT = ENVIRONMENT.upper()

    if ENVIRONMENT == "DOCKER":
        return Settings(
            POSTGRES_HOST="database",  # database image from docker compose
            POSTGRES_PORT=5432,
            POSTGRES_DATABASE=os.getenv("LITSCAN_DB", "reference"),
            POSTGRES_USER=os.getenv("LITSCAN_USER", "docker"),
            POSTGRES_PASSWORD=os.getenv("LITSCAN_PASSWORD", "example"),
            ENVIRONMENT=ENVIRONMENT,
        )
    elif ENVIRONMENT == "LOCAL":
        return Settings(
            POSTGRES_HOST="localhost",
            POSTGRES_PORT=5432,
            POSTGRES_DATABASE="reference",
            POSTGRES_USER="docker",
            POSTGRES_PASSWORD="example",
            ENVIRONMENT=ENVIRONMENT,
        )
    elif ENVIRONMENT == "PRODUCTION":
        return Settings(
            POSTGRES_HOST=os.getenv("POSTGRES_HOST", "192.168.0.6"),
            POSTGRES_PORT=os.getenv("POSTGRES_PORT", 5432),
            POSTGRES_DATABASE=os.getenv("POSTGRES_DATABASE", "reference"),
            POSTGRES_USER=os.getenv("POSTGRES_USER", "docker"),
            POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD", "pass"),
            ENVIRONMENT=ENVIRONMENT,
        )
    elif ENVIRONMENT == "TEST":
        return Settings(
            POSTGRES_HOST="localhost",
            POSTGRES_PORT=5432,
            POSTGRES_DATABASE="test_reference",
            POSTGRES_USER="docker",
            POSTGRES_PASSWORD="example",
            ENVIRONMENT="LOCAL",
        )


def select_examples(conn_str):
    if conn_str is None:
        conn_str = os.getenv("PGDATABASE")
    conn = pg.connect(conn_str)
    cur = conn.cursor()
    cur.execute("select rna_id from litsumm_summaries")
    res = [a[0] for a in cur.fetchall()]
    if len(res) == 0:
        return []
    ids = choices(res, k=10)
    return ids


def search_db(ent_id, conn_str=None):
    if conn_str is None:
        conn_str = os.getenv("PGDATABASE")
    conn = pg.connect(conn_str)
    cur = conn.cursor()
    cur.execute(
        "select * from litsumm_summaries where LOWER(rna_id) = %s", (ent_id.lower(),)
    )
    res = cur.fetchone()
    if res is None:
        (
            context,
            summary,
            cost,
            total_tokens,
            attempts,
            truthful,
            problematic_summary,
            veracity_result,
            selection_method,
        ) = ("Not Found", "Not Found", 0, 0, 0, None, None, "Not Found", "Not Found")
        prompt_1 = "Not Found"
        prompt_2 = "Not Found"
        prompt_3 = "Not Found"
        prompt_4 = "Not Found"
    else:
        (
            s_id,
            _,
            context,
            summary,
            cost,
            total_tokens,
            attempts,
            truthful,
            problematic_summary,
            veracity_result,
            selection_method,
        ) = res
        first_ref = pmcid_pattern.findall(context)[0]
        prompt_1 = context_padding.format(
            ent_id=ent_id, context_str=context, first_ref=first_ref
        )
        prompt_2 = revision_context.format(
            summary=summary, context_str=context, first_ref=first_ref
        )
        prompt_3 = veracity_context.format(
            ent_id=ent_id,
            bullet_summary="- " + summary.replace(". ", "\n- "),
            original_context=context,
        )
        prompt_4 = veracity_revision_context.format(
            checked_assertions=summary, summary=summary
        )

    return (
        summary,
        context,
        total_tokens,
        cost,
        attempts,
        problematic_summary,
        truthful,
        prompt_1,
        prompt_2,
        prompt_3,
        prompt_4,
        veracity_result,
        selection_method,
    )


ENVIRONMENT = os.getenv("ENVIRONMENT", "LOCAL")
if ENVIRONMENT == "LOCAL":
    conn_str = None
else:
    credentials = get_postgres_credentials(ENVIRONMENT)
    conn_str = f"postgresql://{credentials.POSTGRES_USER}:{credentials.POSTGRES_PASSWORD}@{credentials.POSTGRES_HOST}/{credentials.POSTGRES_DATABASE}"

visualisation = gr.Blocks()

with visualisation:
    gr.Markdown(
        "Search for an ID and see the generated summary, and the context from which it was generated."
    )

    with gr.Row():
        id_input = gr.Textbox(label="ID to search for")
        search_button = gr.Button(value="Search")
    with gr.Row():
        examples = gr.Examples(select_examples(conn_str), id_input)

    with gr.Row():
        summary = gr.Textbox(label="Summary")
        context = gr.Textbox(label="Context")
    with gr.Row():
        tokens = gr.Number(label="Tokens", interactive=False)
        cost = gr.Number(label="Cost", interactive=False)
        attempts = gr.Number(label="Attempts", interactive=False)
        with gr.Column():
            problematic = gr.Checkbox(label="Problematic", interactive=False)
            truthful = gr.Checkbox(label="Truthful", interactive=False)
            selection_method = gr.Textbox(label="Selection Method", interactive=False)

    with gr.Row():
        initial_prompt = gr.Textbox(label="Initial Prompt")
        rescue_prompt = gr.Textbox(label="Rescue Prompt")
        veracity_prompt = gr.Textbox(label="Veracity Prompt")
        veracity_rescue_prompt = gr.Textbox(label="Veracity Rescue Prompt")

    with gr.Row():
        veracity_output = gr.Textbox(label="Veracity Output")

    id_input.submit(
        lambda x: search_db(x, conn_str),
        inputs=id_input,
        outputs=[
            summary,
            context,
            tokens,
            cost,
            attempts,
            problematic,
            truthful,
            initial_prompt,
            rescue_prompt,
            veracity_prompt,
            veracity_rescue_prompt,
            veracity_output,
            selection_method,
        ],
    )
    search_button.click(
        lambda x: search_db(x, conn_str),
        inputs=id_input,
        outputs=[
            summary,
            context,
            tokens,
            cost,
            attempts,
            problematic,
            truthful,
            initial_prompt,
            rescue_prompt,
            veracity_prompt,
            veracity_rescue_prompt,
            veracity_output,
            selection_method,
        ],
    )


visualisation.queue(concurrency_count=1)
visualisation.launch(server_name="0.0.0.0", enable_queue=True, server_port=7860)
