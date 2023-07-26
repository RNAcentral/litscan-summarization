#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  \connect $LITSCAN_DB $LITSCAN_USER
  BEGIN;
  CREATE TABLE public.litsumm_summaries (
    id serial PRIMARY KEY,
    rna_id character varying(100),
    context text,
    summary text,
    cost float,
    total_tokens integer,
    problem_summary boolean,
    attempts integer,
    truthful boolean,
    consistency_check_result text,
    selection_method text,
    rescue_prompts text[]
    FOREIGN KEY (rna_id) REFERENCES litscan_job(job_id) ON UPDATE CASCADE ON DELETE CASCADE
  );
  ALTER TABLE public.litsumm_summaries OWNER TO $LITSCAN_USER;
  COMMIT;
EOSQL


psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  \connect $LITSCAN_DB $LITSCAN_USER
  BEGIN;
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
    FOREIGN KEY (summary_id) REFERENCES litsumm_summaries(id) ON UPDATE CASCADE ON DELETE CASCADE
  );
  ALTER TABLE public.litsumm_feedback_single OWNER TO $LITSCAN_USER;
  COMMIT;
EOSQL
