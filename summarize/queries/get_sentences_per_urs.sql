select lsa.pmcid, coalesce(lsbs.sentence, lsas.sentence) as sentence from litscan_result lsr
join litscan_article lsa on lsa.pmcid = lsr.pmcid
join litscan_body_sentence lsbs on lsbs.result_id = lsr.id
left join litscan_abstract_sentence lsas on lsas.result_id = lsr.id


where job_id in ({placeholders})
and not lsa.retracted
and not coalesce(lsbs.sentence, lsas.sentence) like '%%found in an image%%'
