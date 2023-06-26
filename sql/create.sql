CREATE TABLE litsumm_summaries (
id serial PRIMARY KEY,
rna_id text,
context text,
summary text,
cost float,
total_tokens int,
attempt int,
truthful boolean
consistency_check_result text,
selection_method text,
);
