select
xref.upi || '_' || xref.taxid as urs_taxid,
LOWER(split_part(optional_id,'-',1)|| '-' || split_part(optional_id,'-',2) || '-' || split_part(split_part(optional_id,'-',3), '_', 1)   ) as primary_id,
ARRAY[LOWER(split_part(optional_id,'-',2) || '-' || split_part(split_part(optional_id,'-',3), '_', 1)   ),
      LOWER(external_id)
      ] as aliases

from rnc_accessions ac
join xref on xref.ac = ac.accession
join litscan_job lsj on LOWER(split_part(optional_id,'-',1)|| '-' || split_part(optional_id,'-',2) || '-' || split_part(split_part(optional_id,'-',3), '_', 1)   ) = lsj.job_id
left join litscan_job lsj2 on LOWER(split_part(optional_id,'-',2) || '-' || split_part(split_part(optional_id,'-',3), '_', 1)   ) = lsj2.job_id

where ac.database = 'MIRBASE'
and xref.deleted = 'N'
and (lsj.hit_count > 0 or lsj2.hit_count > 0)
