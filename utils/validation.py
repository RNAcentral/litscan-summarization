import re
import logging

pmcid_pattern = re.compile(r'PMC\d+')

def contains_adequate_references(summary:str) -> bool:
    """
    Checks for a ratio of sentence to reference which is good enough. It should ideally be 1:1 but
    sometimes it's ok for it not to be.
    """
    num_sentences = len(summary.split(". ")) # Should be ok?
    ## The following will find references no matter where they are, including if the model dumped them all in one big block
    ## Or decided to number them and stick them at the end. I think for this, that's ok, and we should be able to check they
    ## are in appropriate places some other way
    num_references = len(pmcid_pattern.findall(summary))

    ## this is a somewhat arbitrary threshold, but should make for reasonably well referenced passages
    if num_references / num_sentences > 0.75:
        return True
    logging.warn(f"Reference/sentence ratio was {num_references/num_sentences}, which is below threshold of 0.75")
    
    return False

def references_are_real(summary:str, context:str) -> bool:
    """
    Makes sure all the references in the summary come from the context.
    """
    context_references = set(pmcid_pattern.findall(context))
    summary_references = set(pmcid_pattern.findall(summary))

    ## test if the summary references are a subset of the context ones
    ## Allows for the two sets to be the same, whichI think is ok.
    if summary_references <= context_references:
        return True
    
    ## If the summary is not a subset of the context, the model hallucinated
    logging.warn(f"Model has hallucinated references! The offending items are {summary_references - context_references}")
    return False

def references_end_sentences(summary: str) -> bool:
    """
    The model has a tendency to stick a huge pile of references at the end of the summary, or do other weird
    things. This tries to enforce that any set of square brackets contains a few references, and occurs 
    at the end of each sentence.
    """
    ## this regex allows the model to put more than one reference in a single pair of square brackets
    references_ending_sentences = len(re.findall(r'(\[[PMC0-9,\s]*\].)', summary))
    num_sentences = len(summary.split(". ")) # Should be ok?
    ## Again, bit arbitrary, but as long as half the references are occuring at the end of sentences
    ## the summary is probably ok.
    if references_ending_sentences / num_sentences >= 0.5:
        return True
    logging.warn(f"More than half the references are not found at the end of sentences")
    return False

def not_too_many_refs_per_bracket(summary: str, context: str) -> bool:
    """
    Try to catch cases where the model puts loads of references in one citation. Do it as a percentage of the 
    total number of refs
    """
    references_in_summary = [len(m.split(',')) for m in re.findall(r'(\[[PMC0-9,\s]*\])', summary)]
    num_context_references = len(pmcid_pattern.findall(context))

    if any([ref_block / num_context_references > 0.5 for ref_block in references_in_summary] ):
        logging.warn(f"Model put more than half the available references in a single block!")
        return False
    return True







def validate_summary(summary:str, context:str) :
    """
    Runs a suite of validation functions to ensure the summary is good enough

    Note - this is not going to run any kind of truth checking to make sure the
    summary is fully supported by the context, that needs doing elsewhere.
    """
    checks = {
        "adequate": contains_adequate_references(summary),
        "real" : references_are_real(summary, context),
        "end_sentences": references_end_sentences(summary),
        "appropriate_number": not_too_many_refs_per_bracket(summary, context)
    }

    ## Only give the summary a pass if all check pass
    return checks
