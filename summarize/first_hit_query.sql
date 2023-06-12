select t.primary_id,
        (array_agg(t.result_id)) as result_id,
        (array_agg( DISTINCT t.pmcid)) as pmcid,
        (array_agg( DISTINCT t.job_id)) as job_id,
        (array_agg(t.sentence)) as sentence
        from
        -- sub query to select only the first hit for each article
        (
            select lsb.result_id, max(lsdb.primary_id) as primary_id, (array_agg( DISTINCT lsa.pmcid))[1] as pmcid,
                            (array_agg( DISTINCT lsr.job_id))[1] as job_id,
                            (array_agg(lsb.sentence))[1] as sentence
            from embassy_rw.litscan_body_sentence lsb
            join embassy_rw.litscan_result lsr on lsr.id = lsb.result_id
            join embassy_rw.litscan_database lsdb on lsdb.job_id = lsr.job_id
            join embassy_rw.litscan_article lsa on lsa.pmcid = lsr.pmcid
            where lsdb.name in ('pombase', 'hgnc', 'wormbase', 'mirbase', 'snodb', 'tair', 'sgd', 'pdbe', 'genecards', 'gtrnadb', 'mirgenedb', 'refseq', 'rfam', 'zfin' )
            and lsa.retracted = false
            and lsr.job_id not in ('12s', '12s rrna', '12 s rrna',
                                '13a', '16s', '16s rna', 'rrna', 'e3', 'e2',
                                '16srrna', '16s rrna', 'bdnf', 'nmr',
                                '2a-1', '2b-2', '45s pre-rrna', '7sk',
                                '7sk rna', '7sk snrna', '7slrna', 'rnai',
                                '7sl rna', 'trna', 'snrna', 'mpa', 'msa', 'rns', 'tran',
                                'dmr', 'cta')

            and not lsb.sentence like '%found in an image%'
            and lsdb.primary_id is not NULL
            group by lsb.result_id
        ) as t
    group by t.primary_id
    having cardinality(array_agg(t.result_id)) > 1
