# Nutrition-RAG
Repository for a RAG project with basic nutrition questions

## Methodology
Dense retrieval based on the similarity between the user's question and the previous questions in the data bank. This approach was chosen, because the answers are often short, non-descript and even contain references to the particular syntax of only one of the questions (e.g. polarity markers). The vector database is hosted on Qdrant Cloud.

The dependencies were logged using UV. If you'd like to run the scripts independently, follow the instructions in the Locan Installation section.

TODO

## Setup
1. Make sure you have Docker installed (and Docker Desktop running if using WSL)
2. Set up your env file (see next section about setting up your Qdrant Cloud cluster)
4. Configure the `parameters.toml` file if you'd like, especially if you're
```
docker build --tag=nutritionrag:local .
```
6. Run the docker image with (this might take a minute or two)
```
docker run -it --env-file .env -p 8080:8080 nutritionrag:local
```




## Local Installation (for Swapping Datasets)
1. Install Python (3.10 recommended)
2. Install UV with [your favorite method](https://docs.astral.sh/uv/getting-started/installation/#pypi)
3. Set up a [Qdrant Cloud](https://cloud.qdrant.io) cluster and create an API key for it. If you want to have your cluster locally, replace the `client_source` variable in the `parameters.toml` file with the string `":memory:"`
4. Add the API key to the .env file
5. Set up the dependencies through UV
```
uv sync
```
6. Make sure to set the `force_replace_collection` parameter in the `parameters.toml` file to `"True"` (as a string)
7. Run `rag_pipeline.py` script with
```
python src/nutritionrag/rag_pipeline.py
```
8. Rebuild the docker image with
```
docker build --tag=nutritionrag:local .
```
9. Run the new docker image with (this might take a minute or two)
```
docker run -it --env-file .env -p 8080:8080 nutritionrag:local
```


## Assumptions
### Scope
It is assumed that this is an incomplete FAQ dataset and the pool of users can not be further restricted than humans with diabetes. If the dataset is more specific (e.g. only adults), then the prompt could and should be further tweaked to implement additional failsafes to not respond to questions about children or add caveats about allergies, etc.
### Structure of the dataset
Multiple questions per answer with short answers.
### Language
The language of the questions is assumed to be exclusively English. If that is no longer the case, the mbedding model (and potentially the LLM backend) might need to be replaced.
## Threshold
The current threshold for a retrieved match is set two 0 (i.e. no minimum similarity is required), because based on the preliminary evaluations conducted in `src/nutritionrag/eval.ipynb`, the LLM was actually more likely to default to the fallback option than would have been strictly necessary.

However, this functionality is implemented and ready to use (by changing the `min_similarity_threshold` parameter in the `parameters.toml` file), if this is required in the future.

## Monitoring
Detailed logging (e.g. prometheur & grafana)

These logs can be monitored for latency, outages, etc.
Quality can be monitored by tracking the rate of fallback answers. For example, an increase in the rate of fallback answers could indicate a shift in user queries (a worse fit between the FAQ dataset and the user questions) or a sharp increase could be caused by bugs or outages in the retrieval pipeline.

## Possible expansions in the future
- Hybrid retrieval
- Unit testing
- Quality assessment
- Dedicated API for handling the vector database
