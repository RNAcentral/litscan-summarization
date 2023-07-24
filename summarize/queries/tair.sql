-- select

-- xref.upi || '_' || xref.taxid as urs_taxid,

-- gene as primary,

-- ARRAY[external_id, locus_tag] as aliases


-- from rnc_accessions acc
-- join xref on xref.ac = acc.accession
-- where

-- acc."database" = 'TAIR'

-- and xref.deleted = 'N'
-- ;

select

xref.upi || '_' || xref.taxid as urs_taxid,

gene as primary_id,

ARRAY[external_id, locus_tag] as aliases

-- ,lsj1.hit_count , lsj2.hit_count, lsj3.hit_count

from rnc_accessions acc
join xref on xref.ac = acc.accession
join litscan_job lsj1 on lsj1.job_id = LOWER(gene)
left join litscan_job lsj2 on lsj2.job_id = LOWER(external_id)
left join litscan_job lsj3 on lsj3.job_id = LOWER(locus_tag)
where

acc."database" = 'TAIR'

and xref.deleted = 'N'
and (lsj1.hit_count > 0 OR lsj2.hit_count > 0 OR lsj3.hit_count > 0)
;
