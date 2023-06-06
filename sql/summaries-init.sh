#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  \connect $LITSCAN_DB $LITSCAN_USER
  BEGIN;
  CREATE TABLE public.litsumm_summaries (
    id serial PRIMARY KEY,
    rna_id text,
    context text,
    summary text,
    cost float,
    total_tokens integer
  );
  ALTER TABLE public.litsumm_summaries OWNER TO $LITSCAN_USER;
  COMMIT;
EOSQL
