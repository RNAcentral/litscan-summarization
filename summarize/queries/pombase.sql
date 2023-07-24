select

xref.upi || '_' || xref.taxid as urs_taxid,

gene as primary_id,

string_to_array(gene_synonym, ',') as aliases
from rnc_accessions acc
join xref on xref.ac = acc.accession
where

acc."database" = 'POMBASE'

and xref.deleted = 'N'
;
