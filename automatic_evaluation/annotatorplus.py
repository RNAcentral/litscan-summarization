import re

import click
import polars as pl
import ratelimiter
import requests
from tqdm.auto import tqdm

API_KEY = "41661a9b-68b1-4652-89aa-07c27e0cfb5e"


def clean_text(text):
    """Remove stuff which causes problems with the annotator"""
    ## k-mer where k is a number is problematic for some reason.
    text = re.sub(r"(\d)-", r"\1 ", text)

    ## Add other filters here
    return text


def chunk_context(text, chunk_length=20):
    """Split long contexts into chunks of 20 sentences"""
    chunks = [
        ". ".join(text.split(".")[a * chunk_length : (a * chunk_length) + chunk_length])
        for a in range(int(len(text.split(".")) / chunk_length) + 1)
    ]
    return chunks


@ratelimiter.RateLimiter(max_calls=1, period=1)
def get_annotations(text):
    url = "https://services.data.bioontology.org/annotatorplus"
    res = requests.get(
        url,
        params={
            "apikey": API_KEY,
            "text": text,
            "display_context": "false",
            "display_links": "false",
            "ontologies": "GO",
            "longest_only": "true",
            "exclude_numbers": "true",
            "score": "cvalueh",
            "negation": "true",
            "format": "json",
        },
    )

    if res.ok:
        try:
            annotations = res.json()
            terms = [a["annotatedClass"]["@id"].split("/")[-1] for a in annotations]
            scores = [a["score"] for a in annotations]
        except:
            print("exception")
            print(res.text)
            terms = []
            scores = []
    else:
        print("error")
        terms = []
        scores = []

    return terms, scores


def annotate(row, pbar):
    """
    The function to apply across the dataframe
    returs a dictionary that can be unnested
    """
    summ_terms, summ_scores = get_annotations(clean_text(row["summary"]))
    context_terms = []
    context_scores = []
    for chunk in chunk_context(clean_text(row["context"])):
        terms, scores = get_annotations(chunk)
        context_terms.extend(terms)
        context_scores.extend(scores)

    summ_terms_set = set(summ_terms)
    context_terms_set = set(context_terms)

    missing = context_terms_set - summ_terms_set
    if len(context_terms_set) == 0:
        hit_rate = 0
    else:
        hit_rate = 1 - len(missing) / len(context_terms)

    pbar.update(1)
    return {
        "summary_terms": summ_terms,
        "summary_scores": summ_scores,
        "context_terms": context_terms,
        "context_scores": context_scores,
        "hit_rate": hit_rate,
    }


@click.command()
@click.argument("input_file")
@click.argument("output_file")
@click.option("--feedback_file", default=None)
def main(input_file, output_file, feedback_file=None):
    data = pl.read_parquet(input_file)

    pbar = tqdm(total=len(data))

    data = data.with_columns(
        pl.struct(pl.col("summary"), pl.col("context"))
        .apply(lambda x: annotate(x, pbar))
        .alias("result")
    ).unnest("result")

    data.write_parquet(output_file)


# summary = "MIR944 is a microRNA that has been studied in various contexts. It has been found to be overexpressed in patients with non-small cell lung cancer (NSCLC) [PMC5627048]. MIR944 has also been shown to modulate sensitivity to chemotherapy in other solid tumors [PMC8602334]. The MIR944 promoter is regulated by the transcription factor ΔNp63, which binds to the promoter region and activates transcription [PMC4551945]. The binding of ΔNp63 to the MIR944 promoter is reinforced by the co-regulator AP-2 [PMC4551945]. The chromatin structure of the MIR944 promoter region is relatively open in keratinocytes, which allows for transcription initiation [PMC4551945]. Conservation analyses have shown that MIR944 is a relatively young gene in evolutionary terms, with conservation limited to primates [PMC4551945]. TFAP2A and TFAP2C have been identified as co-regulators that bind to the MIR944 promoter and enhance ΔNp63-mediated activation of transcription [PMC4551945]. The activity of the MIR944 promoter is dependent on ΔNp63 expression, but not linearly dependent on its quantity [PMC4551945]. Overall, these findings suggest that MIR944 has its own independent promoter and its expression is regulated by ΔNp63 and co-regulators such as AP-2, TFAP2A, and TFAP2C."

# context = "), lncRNA (LINC00535, LINC00662, and PTPRN2 lncRNA), and microRNA (MIR4278, MIR1204, MIR944, and MIR921) [PMC9562672]. Another study looked at two specific miRNAs, MIR944 and miR3662, in 85 healthy controls and 90 patients finding that both miRNAs were overexpressed more than fourfold relative to healthy controls in patients with NSCLC [PMC5627048]. 1A); these included MIR1307 and MIR944 that were previously shown to modulate sensitivity to chemotherapy in other solid tumours27,28 and therefore were selected for further validation [PMC8602334]. Supplemental Figure 4D shows that in the top 50 hits of all possible 6-mers, several potential miRNA-binding sites, such as target seeds for mir-374, MIR944, and mir1277, were enriched [PMC4201286]. Among those, miR-186, miR-671, MIR944 and miR-3610 are the most well characterized [PMC8836065]. The activity of the MIR944 promoter was upregulated, whereas that of the ΔNp63 promoter was suppressed (upper panel; Figure 3D) [PMC4551945]. The second-stage PCR was performed with the Nested Universal Primer A (forward primer) and another primary MIR944 transcript-specific primer (reverse primer: 5′-GAGAGGCTGCAGGGAAGAGCAATCT-3′) [PMC4551945]. Interestingly, three consensus p63-binding sites were identified on the genomic sequences of the MIR944 promoter (Figure 3C) [PMC4551945]. Interestingly, we found that ΔNp63-binding to the MIR944 promoter is maintained during the differentiation of keratinocytes (Figure 5A); additionally, mutation of the ΔNp63-binding site blocked the differentiation-induced increase in the activation of the MIR944 promoter (Figure 5B) [PMC4551945]. ChIP-qPCR with anti-TFAP2A and anti-TFAP2C antibodies showed that TFAP2A bound to the MIR944 promoter during the differentiation and that TFAP2C enrichment was increased in differentiated keratinocytes (Figure 5C) [PMC4551945]. Therefore, we hypothesized that TFAP2A and TFAP2C could be co-regulators of the transcriptional targeting of ΔNp63 to the MIR944 promoter during epidermal differentiation [PMC4551945]. Thus, we proposed that co-regulators may function to support the binding of ΔNp63 to the MIR944 promoter, resulting in the maintenance of miR-944 expression [PMC4551945]. Most likely, miR-944 expression is transcriptionally maintained during epidermal differentiation due to reinforcement of the binding of ΔNp63 to the consensus region of the MIR944 promoter by the assistance of its co-regulator AP-2 [PMC4551945]. The open chromatin architecture of the ΔNp63 genomic region in keratinocytes may modulate the initiation of transcription of MIR944 [PMC4551945]. In keratinocytes, the region encompassing the MIR944 promoter has a relatively open chromatin structure, in contrast to the corresponding region in WM266-4 melanoma cells, which do not express ΔNp63 mRNA and miR-944 [PMC4551945]. The p63-binding region of the MIR944 promoter region was visualized using the integrated genome viewer [PMC4551945]. To confirm this, we compared the chromatin status of keratinocytes in the predicted promoter of MIR944 with that of WM266-4 melanoma cells, which do not express miR-944 [PMC4551945]. We further performed site-directed mutagenesis of each of the predicted binding sites in the MIR944 promoter vector to identify the ΔNp63-binding sites more precisely, and found that mutation of the #3 binding site markedly blocked the ΔNp63-mediated activation of the MIR944 promoter (Figure 4G), meaning that ΔNp63 binds to #3 p63-binding site of the MIR944 promoter [PMC4551945]. In this study, we showed that, although miR-944 is generated from TP63-independent transcriptional units, MIR944 transcription is dependent on the transactivating role of ΔNp63 within the promoter region of MIR944 [PMC4551945]. ΔNp63β isoform showed the greatest increase although all isoform of ΔNp63 were effective for the eliciting the activity MIR944 promoter [PMC4551945]. To test this possibility, we first analyzed chromatin signatures that have been established as markers of transcriptionally active promoters in the genomic region of MIR944 [PMC4551945]. To assess binding of ΔNp63 to the MIR944 promoter in keratinocytes, the Illumina Genome Analyzer II platform data set GSE32061 was downloaded from NCBI GEO (http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32061) and analyzed [PMC4551945]. These results indicated that MIR944 is a target of ΔNp63 [PMC4551945]. Furthermore, we identified AP-2 as a co-regulator that reinforces the binding of ΔNp63 to the consensus region of the MIR944 promoter [PMC4551945]. Next, to confirm the functionality of the predicted promoter, we constructed promoter-luciferase vectors containing approximately 2 kb of sequence encompassing either the ΔNp63 or the putative MIR944 promoter (ΔNp63 promoter-luciferase and MIR944 promoter-luciferase, respectively), as illustrated in Figure 3D [PMC4551945]. This analysis allowed us to predict the transcription initiation region, which is located approximately 5 kb downstream of the genomic sequence containing MIR944 (Figure 3A) [PMC4551945]. Moreover, overexpression of ΔNp63 elicited a marked increase in the activity of MIR944 promoter-luciferase (Figure 4D) [PMC4551945]. Moreover, analysis of chromatin state segmentation also predicted a promoter for the independent transcription of MIR944 (Supplementary Figure S7) [PMC4551945]. In order to predict a MIR944 promoter sequence, we used chromatin immunoprecipitation-sequencing (ChIP-Seq) data from human keratinocytes deposited in the UCSC Genome Browser [PMC4551945]. Moreover, analysis of the recent ChIP-Seq study of genome-wide p63-binding in human keratinocytes indicated that p63-binding was enriched at the predicted #3 binding site of the MIR944 promoter region, implying the ΔNp63-mediated activation of this promoter (Figure 4A) (32) [PMC4551945]. As illustrated in our results, the promoter region of MIR944 is active only in ΔNp63 mRNA-expressing cells, but MIR944 promoter activity is not linearly dependent on the quantity of ΔNp63 [PMC4551945]. Conservation analyses using PhyloP in the University of California, Santa Cruz (UCSC) Genome Browser and phylogenetic trees indicated that the nucleotide sequences of the mature miR-944 and the MIR944 stem-loop are only conserved within primates, while the homologues of the TAp63 and ΔNp63 mRNAs are more widely conserved in vertebrates, suggesting that MIR944 is a relatively young gene in the evolutionary hierarchy (Figure 1A and Supplementary Figure S1) [PMC4551945]. Moreover, TFAP2A and TFAP2C depletion reduced the expression of mature miR-944 and the differentiation-induced activation of the MIR944 promoter (Figure 5E and F) [PMC4551945]. The ΔNp63 and MIR944 promoter regions were amplified from human genomic DNA by PCR using the primer sets listed in Supplementary Table S3 [PMC4551945]. In contrast, p63-depletion blocked the differentiation-induced activation of the MIR944 promoter (Figure 4E) [PMC4551945]. Our experimental evidence indicated that miR-944 is generally expressed in cells in which ΔNp63 is specifically expressed, although MIR944 transcription is distinct from the transcription of its host gene, ΔNp63 [PMC4551945]. One possible explanation is that MIR944 is the transcriptional target of the ΔNp63 protein [PMC4551945]. Pan-p63 knockdown diminished ΔNp63-binding to the MIR944 promoter (Figure 4B) and expression of mature miR-944 (Figure 4C) [PMC4551945]. In addition to the open chromatin structure, several co-regulators, including AP-2, may play critical roles in MIR944 transcription and may thus modulate the regulation of miR-944 expression [PMC4551945]. First, we examined whether TFAP2A and TFAP2C bind to the MIR944 promoter during keratinocyte differentiation [PMC4551945]. In this study, we demonstrated that MIR944, which is located in the intron of ΔNp63 has its own promoter: an open chromatin region approximately 5 kb downstream of the MIR944 stem-loop sequence has promoter activity for the transcription of primary MIR944 [PMC4551945]. TFAP2A or TFAP2C overexpression also enhanced the ΔNp63-mediated activation of the MIR944 promoter, as for other p63 target genes, as described in an earlier report (Figure 5D) (32) [PMC4551945]. Here, we suggest that MIR944, a ΔNp63 target gene, could at least partially explain this matter [PMC4551945]. As we had shown that MIR944 contains its own promoter, it was plausible that miR-944 is generated from a separate transcript and not from the host gene TP63 transcript [PMC4551945]. The first-stage of PCR utilized Universal Primer mix (forward primer) and a primary MIR944 transcript-specific primer (reverse primer: 5′-GGGCCTTTATTTGTCTTCCCTGCCA-3′) [PMC4551945]. The ΔNp63 protein directly binds to the MIR944 promoter and drives transcription in keratinocytes, forming a unique regulatory mechanism, in which intronic miRNA is generated through the transcriptional activity of its host gene [PMC4551945]. Next, the 5′-end of primary MIR944 transcripts was determined by 5′RACE using a SMARTer RACE cDNA Amplification kit (Clontech, Mountain View, CA, USA), followed by a nested PCR reaction [PMC4551945]. Intron 4 of human TP63 contains the gene for miR-944 (MIR944) [PMC4551945]. Moreover, TSA treatment increased the activity of the ΔNp63 promoter, but not that of the MIR944 promoter (bottom panel, Figure 3D) [PMC4551945]. showed an overlap between two of the tissue sources:SLC25A3 and ANGPT1 in BM- and WAT-STCs;MIR326, ITGB8, CFB, BMP4, CFI, SEPP1, SVEP1, ANKRD29, GUCY1B3, YY2 and TPTE2P6 in BM- and UC-STCs andCLEC18A, ARHGEF3, SPN, MIR944, YME1L1 and LOC101928303 in UC- and WAT-STCs.These findings are also reflected by a heatmap (Fig [PMC6510770]."
# # context = context.replace(";", "")


# summ_terms, summ_scores = get_annotations(summary)

# ## Context has to be handled a bit different - per sentence
# context_terms = []
# context_scores = []
# chunk_length = 20
# chunks = [". ".join(context.split('.')[a*chunk_length:(a*chunk_length)+chunk_length]) for a in range(int(len(context.split('.'))/chunk_length) + 1 )]
# print(chunks, len(chunks))
# for chunk in chunks:
#     print(chunk)
#     terms, scores = get_annotations(chunk)
#     context_terms.extend(terms)
#     context_scores.extend(scores)

# print(context_terms)
# print(context_scores)
# print(summ_terms)
# print(summ_scores)

# context_terms = set(context_terms)
# summ_terms = set(summ_terms)

# print(context_terms)
# print(summ_terms)
# missing = context_terms - summ_terms
# print(missing)
# print( 1 - len(missing)/len(context_terms))


if __name__ == "__main__":
    main()
