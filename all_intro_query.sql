select result_id, lsa.pmcid as pmcid,
				lsr.job_id as job_id,
				sentence
from embassy_rw.litscan_body_sentence lsb
join embassy_rw.litscan_result lsr on lsr.id = lsb.result_id
join embassy_rw.litscan_database lsdb on lsdb.job_id = lsr.job_id
join embassy_rw.litscan_article lsa on lsa.pmcid = lsr.pmcid

where location = 'intro'
where name in ('pombase', 'hgnc', 'wormbase', 'mirbase', 'snodb', 'tair', 'sgd', 'pdbe', 'genecards', 'gtrnadb', 'mirgenedb', 'refseq', 'rfam', 'zfin' )
and retracted = false
and lsr.job_id not in ('12s', '12s rrna', '12 s rrna',
                       '13a', '16s', '16s rna',
                       '16srrna', '16s rrna',
                       '2a-1', '2b-2', '45s pre-rrna', '7sk',
                       '7sk rna', '7sk snrna', '7slrna',
                       '7sl rna', 'trna', 'snrna', 'mpa', 'msa', 'rns', 'tran')
