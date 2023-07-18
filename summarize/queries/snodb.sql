select

xref.upi || '_' || xref.taxid as urs_taxid,

split_part(split_part(description, ')', 1), '(', 2) as primary,
ARRAY[''] as aliases

from rnc_accessions acc
join xref on xref.ac = acc.accession
where

acc."database" = 'SNODB'

and xref.deleted = 'N'
;
