FROM postgres:15.2-alpine
RUN wget https://raw.githubusercontent.com/RNAcentral/rnacentral-references/main/database/scripts/init.sh \
    && mv init.sh /docker-entrypoint-initdb.d/
COPY sql/summaries-init.sh /docker-entrypoint-initdb.d/