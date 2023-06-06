## LitSumm - summarize literature with LLMs

### What is this
The single entrypoint for the LitScan + LitSumm literature scanning and summarization tool. With the stuff here, you should be able to search for an RNA and generate summaries from the literature.

### How do I use it

You should be able to just run `docker compose -f docker-compose-{mac,x86}.yml up --build` in this repo to have a functioning LitScan + LitSumm working locally.

Some of the dependencies don't play well with the emulation on Apple silicon, so there is a separate compose file for each architecture.

#### Preparation
You will need a .env file in this directory with the following contents:

1. ENVIRONMENT=docker
2. LITSCAN_USER=litscan_user
3. LITSCAN_PASSWORD=any_pass
4. LITSCAN_DB=litscan_db
5. DEVICE=cpu:0
6. MODEL_NAME=chatGPT
7. TOKEN_LIMIT=2560
8. OPENAI_API_KEY=<your api key here>

Some notes:
- DEVICE can also be `mps` or `cuda` if you have those available.
- MODEL_NAME should always be chatGPT for now - in future this is where we will make the pipeline use local models
- TOKEN_LIMIT could be a bit higher, but this is the acceptable safety margin so that responses are not truncated. I would not set this higher than 3072.
- To get an openAI API key, sign up here: https://platform.openai.com/account/api-keys

To launch, run `docker compose -f docker-compose-mac.yml up --build` (substitute the x86 compose file if you're not on an M1/M2 mac)

Once everything is up and running, you should see messages about polling for new jobs. This mean's you're ready to go.

To submit an ID for processing, you do `curl -H "Content-Type:application/json" -d "{\"id\": \"bta-mir-16a\"}" localhost:8080/api/submit-job` which sends the request to the LitScan API. LitScan will then search for articles mentioning the ID (in this example bta-mir-16a) and put the results in the database, from where LitSumm will pick them up, run the summarization chain and place the summary into the database's `litsumm_summaries` table.

To connect to the database, you can use a string that looks something like this: `psql postgres://litscan_user:any_pass@localhost:5432/litscan_db`

For the above ID, the end-to-end processing time from submission to summary being saved in the database is ~2 minutes, and the cost is about $0.008. The summary produced looks like this:

    Bta-mir-16a is a miRNA that has been studied in various contexts. It has been quantified in oxidized RNA solutions and used as an internal control to normalize the read counts of other miRNAs [PMC9866016]. It has also been found to be significantly up-regulated in the summer in comparison to Sahiwal [PMC9686552]. Bta-mir-16a is one of five miRNAs related to bovine mastitis inflammation and targets four differentially expressed genes [PMC8511192]. In B cell chronic lymphocytic leukemia, the expression of bta-mir-16a is down-regulated, but it was found to be the most stable miRNA in B cells among all 22 cattle, both BLV-infected and BLV-uninfected cattle [PMC8432782]. Bta-mir-16a was also one of the most abundant miRNAs identified among 510 mature miRNAs, with bta-miR-21-5p being the most highly expressed [PMC6600136]. In addition, bta-mir-16a was found to be significantly upregulated in mastitis-affected dairy cows compared with healthy cows [PMC6107498]. Finally, bta-mir-16a was differentially regulated by SFO and had high intra-modular connectivity suggesting involvement in regulating traits that were significantly correlated with the turquoise module in SFO treatment [PMC6164576].
