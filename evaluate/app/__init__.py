import os
import random
import uuid
from collections import namedtuple

import psycopg2
from flask import Flask, make_response, redirect, render_template, request, url_for
from flask_bootstrap import Bootstrap5

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


ENVIRONMENT = os.getenv("ENVIRONMENT", "LOCAL")
if ENVIRONMENT == "LOCAL":
    conn_str = os.getenv("PGDATABASE")
else:
    credentials = get_postgres_credentials(ENVIRONMENT)
    conn_str = f"postgresql://{credentials.POSTGRES_USER}:{credentials.POSTGRES_PASSWORD}@{credentials.POSTGRES_HOST}/{credentials.POSTGRES_DATABASE}"

app = Flask(__name__)
Bootstrap5(app)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def intro():
    resp = make_response(render_template("intro.html"))
    return resp


@app.route("/search")
def search():
    resp = make_response(render_template("search_and_show.html"))
    return resp


@app.route("/reset_seen")
def reset_seen():
    ent_id = request.args.get("ent_id")
    resp = make_response(url_for("intro"))
    resp.set_cookie("seen_ids", "")
    return resp


@app.route("/single")
def present_single_summary():
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()

    ## Get the IDs this browser has seen
    if request.cookies.get("seen_ids") == "" or request.cookies.get("seen_ids") is None:
        seen_ids = []
    else:
        seen_ids = [int(i) for i in request.cookies.get("seen_ids").split(" ")]
    print(seen_ids)
    ## Get maximum ID
    cur.execute("SELECT COUNT(id) FROM litsumm_summaries;")
    N = cur.fetchone()[0]
    cur.execute("SELECT MIN(id) FROM litsumm_summaries;")
    first_id = cur.fetchone()[0]

    ## Query to randomly select a single summary
    if len(seen_ids) == 0:
        selected = first_id
    else:
        selected = seen_ids[-1] + 1

    print(seen_ids, selected, N)

    if selected is None or first_id is None:
        placeholder = "Nothing in the database yet!"
        resp = make_response(
            render_template(
                "single_summary.html",
                summary=placeholder,
                rna_id=placeholder,
                context=placeholder,
                summ_id=-1,
            )
        )
    else:
        if (selected - first_id) >= N:
            selected = first_id

        if not selected in seen_ids:
            seen_ids.append(selected)
        print(selected)
        QUERY = f"SELECT * FROM litsumm_summaries WHERE id = {selected};"
        cur.execute(QUERY)
        (
            summ_id,
            rna_id,
            context,
            summary,
            cost,
            total_tokens,
            attempts,
            truthful,
            problematic_summary,
            veracity_result,
            selection_method,
        ) = cur.fetchone()

        app.logger.debug(rna_id)
        resp = make_response(
            render_template(
                "single_summary.html",
                summary=summary,
                rna_id=rna_id,
                context=context,
                summ_id=summ_id,
            )
        )

        resp.set_cookie("seen_ids", " ".join([str(i) for i in seen_ids]))

    user_id = request.cookies.get("user")
    if user_id is None:
        user_id = uuid.uuid4()
        resp.set_cookie("user", str(user_id))

    return resp


@app.route("/save_single", methods=["GET", "POST"])
def save_single_feedback():
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()

    feedback = request.json

    user_id = request.cookies.get("user")
    feedback["user_id"] = str(user_id)

    print(feedback)

    cur.execute(
        """INSERT INTO litsumm_feedback_single(user_id, summary_id, feedback,
                                contains_hallucinations, inaccurate_text,
                                contradictory, over_specific, bad_length,
                                mentions_ai, short_context, false_positive, free_feedback) VALUES (
        %(user_id)s, %(summary_id)s, %(feedback)s, %(contains_hallucinations)s,
        %(inaccurate_text)s, %(contradictory)s, %(over_specific)s, %(bad_length)s,
        %(mentions_ai)s, %(short_context)s, %(false_positive)s, %(free_feedback)s
        )""",
        feedback,
    )

    conn.commit()

    return url_for("present_single_summary")
