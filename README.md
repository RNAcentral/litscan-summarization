## LitSumm - summarize literature with LLMs

### What is this
The single entrypoint for the LitScan + LitSumm literature scanning and summarization tool. With the stuff here,
you should be able to search for an RNA (or anything else) and generate summaries from the literature.

### How do I use it

You should be able to just run `docker compose -f docker-compose-{mac,x86}.yml up --build` in this repo to have a
functioning LitScan + LitSumm working locally.

Some of the dependencies don't play well with the emulation on Apple silicon, so there is a separate compose file
for each architecture.

#### Preparation
You will need a `.env` file in this directory with the following contents:
```
ENVIRONMENT=docker
LITSCAN_USER=litscan_user
LITSCAN_PASSWORD=any_pass
LITSCAN_DB=litscan_db
DEVICE=cpu:0
MODEL_NAME=chatGPT
TOKEN_LIMIT=2560
OPENAI_API_KEY=<your api key here>
```

Some notes:
- You can use any value for LITSCAN_USER, LITSCAN_PASSWORD and LITSCAN_DB
- DEVICE can also be `mps` or `cuda` if you have those available.
- MODEL_NAME should always be chatGPT for now - in future this is where we will make the pipeline use local models
- TOKEN_LIMIT could be a bit higher, but this is the acceptable safety margin so that responses are not truncated. I would not set this higher than 3072.
- To get an openAI API key, sign up here: https://platform.openai.com/account/api-keys

To launch, run `docker compose -f docker-compose-mac.yml up --build` (substitute the x86 compose file if you're not on an M1/M2 mac)

Once everything is up and running, you should see messages about polling for new jobs. This mean's you're ready to go.

#### Submit a single job

To submit an ID for processing, you do
```
curl -H "Content-Type:application/json" -d "{\"id\": \"bta-mir-16a\"}" localhost:8080/api/multiple-jobs
```
which sends the request to the LitScan API. LitScan will then search for articles mentioning the ID
(in this example `bta-mir-16a`) and put the results in the database, from where LitSumm will pick them up,
run the summarization chain and place the summary into the database's `litsumm_summaries` table.

To connect to the database, you can use a string that looks something like this:
```
psql postgres://litscan_user:any_pass@localhost:8082/litscan_db
```

For the above ID, the end-to-end processing time from submission to summary being saved in the database is ~2 minutes,
and the cost is about $0.008. The summary produced looks like this:
```
Bta-mir-16a is a miRNA that has been studied in various contexts. It has been quantified in oxidized RNA solutions and
used as an internal control to normalize the read counts of other miRNAs [PMC9866016]. It has also been found to be
significantly up-regulated in the summer in comparison to Sahiwal [PMC9686552]. Bta-mir-16a is one of five miRNAs
related to bovine mastitis inflammation and targets four differentially expressed genes [PMC8511192]. In B cell chronic
lymphocytic leukemia, the expression of bta-mir-16a is down-regulated, but it was found to be the most stable miRNA in
B cells among all 22 cattle, both BLV-infected and BLV-uninfected cattle [PMC8432782]. Bta-mir-16a was also one of the
most abundant miRNAs identified among 510 mature miRNAs, with bta-miR-21-5p being the most highly expressed
[PMC6600136]. In addition, bta-mir-16a was found to be significantly upregulated in mastitis-affected dairy cows
compared with healthy cows [PMC6107498]. Finally, bta-mir-16a was differentially regulated by SFO and had high
intra-modular connectivity suggesting involvement in regulating traits that were significantly correlated with the
turquoise module in SFO treatment [PMC6164576].
```

Be aware that **some IDs can have a huge amount of articles to parse, which can take a long time to run**.
To avoid this problem, you can set a maximum number of articles to be searched by LitScan as follows
```
curl -H "Content-Type:application/json" -d "{\"id\": \"NEAT1\", \"search_limit\": 100}" localhost:8080/api/multiple-jobs
```

Another important detail about LitScan is that **by default the following query is used**
```
query=("ID" AND ("rna" OR "mrna" OR "ncrna" OR "lncrna" OR "rrna" OR "sncrna"))
```

Where:
1. `"ID"` is the string used in the search
2. `("rna" OR "mrna" OR "ncrna" OR "lncrna" OR "rrna" OR "sncrna")` is used to filter out possible false positives

If you want to run a non-RNA related job, use the `query` parameter to change the query
```
curl -H "Content-Type:application/json" -d "{\"id\": \"P2_Phage_GpR\", \"query\": \"('protein' AND 'domain')\"}" localhost:8080/api/multiple-jobs
```

Or if you don't want to use a query to filter out potential false positives, run
```
curl -H "Content-Type:application/json" -d "{\"id\": \"P2_Phage_GpR\", \"query\": \"\"}" localhost:8080/api/multiple-jobs
```

To rescan an id and create a new summary, use the `rescan` parameter
```
curl -H "Content-Type:application/json" -d "{\"id\": \"bta-mir-16a\", \"rescan\": true}" localhost:8080/api/multiple-jobs
```

#### Submit multiple jobs

To submit multiple IDs for processing, you do
```
curl -H "Content-Type:application/json" -d "{\"id\": \"RF00016\", \"job_list\": [\"SNORD14\", \"U14A\", \"U14 snoRNA\"], \"search_limit\": 20}" localhost:8080/api/multiple-jobs
```
which sends the request to the LitScan API. LitScan will create a job for each ID (in this example `RF00016`, 
`SNORD14`, `U14A` and `U14 snoRNA`) and then search for articles mentioning the ID. Results will be saved in the 
database, from where LitSumm will pick them up, run the summarization chain and place the summary into the 
database's `litsumm_summaries` table.

When submitting multiple IDs, use the `id` field for submitting the accession and the `job_list` field for other 
names/synonyms of this accession. The summary will be created based on the `id` and `job_list` sentences.

### Visualising the results

You can visualise your summaries by going to [http://localhost:7860](http://localhost:5000) which will show a Gradio app pulling data from the database inside LitSumm.

Here, you will be able to search the database for IDs you have run and populate the fields. Briefly, they are:

- Summary: The summary generated by LitSumm after all validation and processing.
- Context: The context LitSumm used to generate the summary.
- Tokens: The total number of tokens used. This isn't the number on the context/summary, but the sum of all requests, so if I sent a 500 token context to be summarised, got a 100 token reply and checked it for consistency (uses ~200-300 tokens), I would probably see something like 1000 tokens here.
- Cost: The const in dollars of producing this summary.
- Attempts: The number of retries used. If this is 1, we got the summary right on the first go. It can be up to 4 if rewrites are needed.
- Problematic: If the automated rescue prompt doesn't work after 4 tries, we mark the summary as problematic.
- Truthful: This is checked if the consistency check is passed first time. If it isn't checked, then we will have run the consistency based amendment prompt.
- Initial Prompt: The context wrapped into the prompt, so you can see how big it was.
- Rescue Prompt: The context and summary formatted into the revision prompt we tried.
- Veracity Prompt: The self-consistency check we got the LLM to run.
- Veracity Rescue Prompt: if inconsistencies are found, we run this to try and fix them
- Veracity Output: The response from the LLM telling us which statements were true and which were false.

Some important things to note however:

- The ID must match exactly! This means no trailing whitespace.
- Depending on how you initialise the visualisation, you may or may not actually get examples. If there is nothing in the DB, there will be no examples.
- It isn't possible at the moment to do any kind of autocomplete, so you will need to lookup IDs some other way



### Evaluating the results

There is another interface for evaluation, which will run at [http://localhost:5000](http://localhost:5000). This interface should allow you to quickly iterate on many IDs and give feedback.

We show the context in a collapsible text field above the summary, with the entity ID as the title. Prompts are also shown at the bottom.

The primary feedback is a 1-5 scale of how good you think the summary is, with 5 being great and 1 being useless. There is additional feedback to collect stats on the most egregious errors. This is accessed from the additional feedback dropdown below the summary and comprises some checkboxes and a free text entry.

To submit your feedback, click submit. To move to another summary, click next or previous. The evaluator will run through summaries in the order they are in the database, so should be deterministic for each user.

**The tool will store some cookies on your browser**

There are two: a name, so we can disambiguate feedback based on user; and another that simply stores a comma separated list of the IDs you have seen, so we can pick up where you left off and navigate forward and back easily.

Feedback is stored in the same database as everything else, in the `litsumm_feedback` table.


## Troubleshooting

### I see warnings about numba

This is to do with the implementation of UMAP used in the topic modelling being based on an older version of numba
in which there are some deprecated features. It will still work, and at some point the package will be updated and
they'll go away.

### I get an error about duplicate keys in litscan_consumer
This happens because the network in docker changes between runs. One way to fix it is to delete the volume associated
with the containers, but this will also nuke your summaries if you've already made some.

Better is to do the following:

1. `docker compose -f docker-compose-mac.yml up database` to bring the database up
2. `psql postgres://litscan_user:any_pass@localhost:8082/litscan_db` to connect to the database
3. `TRUNCATE TABLE litscan_consumer` to clear out the table and start again
