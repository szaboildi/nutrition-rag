# nutrition-rag
Repository for a RAG project with basic nutrition questions

## Installation
install uv
make sure you have docker installed (and running)
set up your env file

## Assumptions
Language
It is assumed that this is an incomplete FAQ dataset and the pool of users can not be further restricted than humans with diabetes. If the dataset is more specific (e.g. only adults), then the prompt could and should be further tweaked to implement additional failsafes to not respond to questions about children or add caveats about allergies, etc.

## Caveats

## Monitoring
detailed loggins (e.g. prometheur & grafana)

logs can be monitored for latency, outages, etc.
Quality can be monitored by tracking the rate of fallback answers. For example, an increase in the rate of fallback answers could indicate a shift in user queries (a worse fit between the FAQ dataset and the user questions) or a sharp increase could be caused by bugs or outages in the retrieval pipeline.
