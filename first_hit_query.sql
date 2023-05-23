select result_id, (array_agg( DISTINCT lsa.pmcid))[1] as pmcid,
				(array_agg( DISTINCT lsr.job_id))[1] as job_id,
				(array_agg(sentence))[1] as sentence
from embassy_rw.litscan_body_sentence lsb
join embassy_rw.litscan_result lsr on lsr.id = lsb.result_id
join embassy_rw.litscan_database lsdb on lsdb.job_id = lsr.job_id
join embassy_rw.litscan_article lsa on lsa.pmcid = lsr.pmcid
where name in ('pombase', 'hgnc', 'wormbase', 'mirbase', 'snodb', 'tair', 'sgd', 'pdbe', 'genecards', 'gtrnadb', 'mirgenedb', 'refseq', 'rfam', 'zfin' )
and retracted = false
and lsr.job_id not in ('12s', '12s rrna', '12 s rrna',
                       '13a', '16s', '16s rna', 'rrna', 'e3', 'e2',
                       '16srrna', '16s rrna', 'bdnf', 'nmr',
                       '2a-1', '2b-2', '45s pre-rrna', '7sk',
                       '7sk rna', '7sk snrna', '7slrna', 'rnai',
                       '7sl rna', 'trna', 'snrna', 'mpa', 'msa', 'rns', 'tran',
                       'mir-21', 'mir-155')
group by result_id

having cardinality(array_agg(lsb.id)) > 2 and cardinality(array_agg(DISTINCT lsr.job_id)) = 1
