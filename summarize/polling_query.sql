select result_id, max(primary_id) as primary_id, (array_agg( DISTINCT lsa.pmcid))[1] as pmcid,
				(array_agg( DISTINCT lsr.job_id))[1] as job_id,
				(array_agg(sentence))[1] as sentence
from embassy_rw.litscan_body_sentence lsb
join embassy_rw.litscan_result lsr on lsr.id = lsb.result_id
join embassy_rw.litscan_database lsdb on lsdb.job_id = lsr.job_id
join embassy_rw.litscan_article lsa on lsa.pmcid = lsr.pmcid
where name in ('pombase', 'hgnc', 'wormbase', 'mirbase', 'snodb', 'tair', 'sgd', 'pdbe', 'genecards', 'gtrnadb', 'mirgenedb', 'refseq', 'rfam', 'zfin' )
and retracted = false
and lsr.job_id in %s
and not sentence like '%found in an image%'
and primary_id is not NULL
group by result_id

-- having cardinality(array_agg(lsb.id)) > 2 and cardinality(array_agg(DISTINCT lsr.job_id)) = 1
