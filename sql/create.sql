CREATE TABLE litsumm_summaries (
id serial PRIMARY KEY,
rna_id text,
context text,
summary text,
cost float,
total_tokens int,
attempt int,
problem_summary boolean,
truthful boolean
consistency_check_result text,
selection_method text,
);

CREATE TABLE litsumm_feedback_single (
    feedback_id serial PRIMARY KEY,
    user_id text,
    summary_id int,
    feedback int,
    contains_hallucinations boolean,
    inaccurate_text boolean,
    contradictory boolean,
    over_specific boolean,
    bad_length boolean,
    mentions_ai boolean,
    short_context boolean,
    false_positive boolean,
    free_feedback text,
    FOREIGN KEY (summary_id) REFERENCES litsumm_summaries(id)
);
