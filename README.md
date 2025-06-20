# Nutrition-RAG
Repository for a RAG project with basic nutrition questions. This functionality could be offered as a part of a bigger app for people with diabetes, where users could ask questions based on a more expanded FAQ. The FAQ could be compiled by previous interactions with nutritionists and could be regularly expanded with more user questions over time (with the appropriate data permissions). In case no answers can be given to the user, the user could be redirected to other credible sources or experts. Ideally, patients could give feedback on the responses they got, which could be collected for quality monitoring purposes (see below).

The pipeline contains five components:
1. Preprocessing (happens within `src/nutritionrag/preprocessing.py`)
2. Retrieval (happens within `src/nutritionrag/rag_pipeline.py`):
Retrieves question-answer pairs based on the similarity between the user's question and the previous questions in an FAQ dataset (cosine similarity with dense embeddings using the `intfloat/e5-base` model). Since the dataset contains five sample questions for each answer, the same answer can be retrieved multiple times (just with different questions). The current retrieval limit (k) is set to 5 by default (in `parameters.toml`).
The question-to-question similarity approach was chosen because the answers are often short, non-descript and even contain references to the particular syntax of only one of the questions (e.g. polarity markers). With this limited dataset the questions offer more variation. The default vector database is hosted on Qdrant Cloud.
3. Filtering of the results (currently inactive with threshold set to -1; happens within `src/nutritionrag/rag_pipeline.py`):
The current threshold for a retrieved match is set to -1 (i.e. no minimum similarity is required), because based on the preliminary evaluations conducted in `src/nutritionrag/eval.ipynb`, the LLM was actually more likely to default to the fallback option than would have been strictly necessary.
However, this functionality is implemented and ready to use (by changing the `min_similarity_threshold` parameter in the `parameters.toml` file), if this is required in the future.
4. Generation (or Fallback response): At most five question-answer pairs are passed on to the LLM (`gpt-4o` by default). The LLM is instructed with a system prompt about the task, including to give a fallback answer if the question cannot be answered based on the documents that were retrieved.
If no question-answer pairs meet the minimum cosine similarity threshold (currently set to -1, so all hits are automatically passed on), then the LLM-based generation is skipped and a fallback answer is returned to the user.
5. App: The final answer (whether LLM-generated or the filtering fallback response) is then displayed in a streamlit app (defined at `src/streamlit/app.py`) along with the 3 best-matching unique answers (and their best matching question).

The app is dockerized and can be run locally following the setup instructions outlined below.
The dependencies were logged using UV. If you'd like to run the scripts independently, follow the instructions in the Local Installation section. There is a small set of 15 hand-annotated evaluation questions to test the model performance with at `data/eval/test_questions_raw.csv`.


## Setup
1. Make sure you have Docker installed (and Docker Desktop running if using WSL).
2. Clone this repository
3. Create a `.env` file based on `.env.sample`. Replace the values with your Qdrant Cloud and OpenAI API keys.
4. Load your `.env` file
```
source .env
```
5. At this point, you can confirm whether you are happy with the default configuration in `parameters.toml`. If not, change the parameters.
6. Build the docker image (this takes about 5-10 minutes) with
```
docker build --tag=nutritionrag:local .
```
7. Run the docker image (this might take a minute or two until the interface appears) with
```
docker run -it --env-file .env -p 8080:8080 nutritionrag:local
```
8. You can open the app in your browser at `http://0.0.0.0:8080`


## Local Installation (for Swapping Datasets)
1. Install UV with [your favorite method](https://docs.astral.sh/uv/getting-started/installation/#pypi)
2. Set up a [Qdrant Cloud](https://cloud.qdrant.io) cluster and create an API key for it. If you want to have your vector database hosted locally, replace the `client_source` variable in the `parameters.toml` file with the string `":memory:"`.
3. Add the Qdrant Cloud API key to the `.env` file
4. Reload your `.env` file
```
source .env
```
5. Create a virtual environment
```
uv venv .venv
```
6. Activate the virtual environment
```
source .venv/bin/activate
```
7. Set up the dependencies through UV
```
uv sync
```
8. Create the folder for the local raw input files
```mkdir data/raw_input_files```
9. Move your input files to this new folder. They have to be `.json` files containing lists of question-list + answer dictionaries:
```[
  {
    "questions": [
      "Question1a",
      "Question1b",
      ...
    ],
    "answer": "Answer1"
  },
  {...}
]
```
10. Make sure to set the `force_replace_collection` parameter in the `parameters.toml` file to `"True"` (as a string) and that the `client_source` parameter points to either your local device (`":memory:"`) or a Qdrant Cloud cluster that you have writing access to. Make sure that `collection_name` is set to a name of your chosing (if it already exists, it will be overwritten).
11. Run the `rag_pipeline.py` script with
```
uv run src/nutritionrag/rag_pipeline.py
```
12. Rebuild the docker image (this takes about 5-10 minutes) with
```
docker build --tag=nutritionrag:local .
```
13. Run the new docker image (this might take a minute or two until the interface appears) with
```
docker run -it --env-file .env -p 8080:8080 nutritionrag:local
```

## Performance
The RAG pipeline gave an acceptable solution 10/15 times (66% accuracy) in the (non-representative) hand-annotated dataset created for this project (included in the repo at `data/eval/test_questions_raw.csv`). All 5 mistakes are due to the LLM returning false negatives (the fallback response when it wouldn't necessarily have to).

In further steps this can be mitigated by increasing the original dataset to include question-answer pairs that have a more abstract relationship.

As an alternative to the question-to-question similarity search, a question to Q&A search was also briefly explored, but this did not yield substantial improvements in performance.


## Assumptions

### Scope
It is assumed that this is an incomplete FAQ dataset and the pool of users can not be further restricted than humans with diabetes. If the dataset is more specific (e.g. only adults), then the prompt could and should be further tweaked to implement additional failsafes to not respond to questions about children or add caveats about allergies, etc.

### Structure of the dataset
Datasets come in a `.json` format, with multiple questions per answer and the answers are typically short, often not matching the natural language features of all of the questions that they are paired with.

### Language
The language of the questions is assumed to be exclusively English. If that is no longer the case, the mbedding model (and potentially the LLM backend) might need to be replaced. The current setup only suports full deletion and recreation.


## Possible expansions in the future

### Hybrid retrieval

### Unit testing

### Detailed Quality assessment

### Dedicated API for handling the vector database
A dedicated API could be developed to process updates in the database automatically. This could cover files being removed from the database or partially updated.

### Monitoring:
1. Detailed logging (e.g. Prometheus & Grafana):
These logs can be monitored for latency, outages, etc.
Quality can be monitored by tracking the rate of fallback answers. For example, an increase in the rate of fallback answers could indicate a shift in user queries (a worse fit between the FAQ dataset and the user questions) or a sharp increase could be caused by bugs or outages in the retrieval pipeline.
2. User feedback: Another option would be to regularly monitor user feedback to identify either shifts in usage patterns or patterns that were otherwise missed during development.
