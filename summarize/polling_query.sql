select t.primary_id,
		(array_agg(t.result_id)) as result_id,
		(array_agg(t.pmcid)) as pmcid,
      (array_agg(t.job_id)) as job_id,
		(array_agg(t.sentence)) as sentence

	from(
	-- subquery selects only first hit from each article
    select lsb.result_id,
            (array_agg( DISTINCT lsa.pmcid))[1] as pmcid,
            (array_agg( DISTINCT lsr.job_id))[1] as primary_id,
                (array_agg( DISTINCT lsr.job_id))[1] as job_id,
            (array_agg(lsb.sentence))[1] as sentence
    from litscan_body_sentence lsb
    join litscan_result lsr on lsr.id = lsb.result_id
    join litscan_article lsa on lsa.pmcid = lsr.pmcid
    and lsa.retracted = false
    and lsr.job_id in ({placeholders})
    and not lsb.sentence like '%%found in an image%%'
    group by lsb.result_id
    ) as t
    group by t.primary_id
