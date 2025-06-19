import glob
import os

from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

from nutritionrag.preprocessing import process_text

import argparse
from openai import OpenAI



def setup_vector_db(
    encoder_name:str="intfloat/e5-base", sparse_retriever:str="BM25",
    client_source:str=":memory:",
    qdrant_cloud_api_key:str="None", from_scratch:bool=False,
    input_folder:str="data/raw_input_files",
    collection_name:str="dummy_name", dist_name:str="COSINE",
    input_folder_qa:str="data",
    relevance_score_file_prefix:str="sample_qa_passage_lvl",
    sample_qa_file:str="sample_qa.json", hybrid:bool=False):

    if not hybrid:
        encoder = SentenceTransformer(encoder_name)
        if client_source == ":memory":
            client = QdrantClient(client_source)
        else:
            client = QdrantClient(
                url=client_source, api_key=qdrant_cloud_api_key)

        if not from_scratch:
            print("Vector database loaded")
            return client, encoder

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=getattr(models.Distance, dist_name.upper()),
            ),
        )

        file_names = glob.glob(f"{input_folder}/*")
        for file_name in file_names:
            # Set up the passages
            input_passages_dict = process_text(
                path=file_name, input_folder_qa=input_folder_qa,
                relevance_score_file_prefix=relevance_score_file_prefix,
                sample_qa_file=sample_qa_file)

            print(f"Text cleaned in {file_name}")

            client.upload_points(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=idx, vector=encoder.encode(
                            doc["question"], normalize_embeddings=True).tolist(),
                        payload=doc)
                    for idx, doc in enumerate(input_passages_dict)
                ],
            )

        print("Vector database created")
        return client, encoder

    # Hybrid retrieval
    # Set up encoder and client
    if client_source == ":memory":
            client = QdrantClient(client_source)
    else:
        client = QdrantClient(
            url=client_source, api_key=qdrant_cloud_api_key)

    if not from_scratch:
        print("Vector database loaded")
        return client, encoder

    dense_embedding_model = TextEmbedding("intfloat/multilingual-e5-large")
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")
    late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

    file_names = glob.glob(f"{input_folder}/*")
    for file_name in file_names:
        # Set up the passages
        input_passages_dict = process_text(
            path=file_name, input_folder_qa=input_folder_qa,
            relevance_score_file_prefix=relevance_score_file_prefix,
            sample_qa_file=sample_qa_file)

        print(f"Text cleaned in {file_name}")

    documents = [i["question"] for i in input_passages_dict]

    dense_embeddings = list(dense_embedding_model.embed(doc for doc in documents))
    bm25_embeddings = list(bm25_embedding_model.embed(doc for doc in documents))
    late_interaction_embeddings = list(late_interaction_embedding_model.embed(doc for doc in documents))

    client.create_collection(
        "hybrid-search",
        vectors_config={
            "multilingual-e5-large": models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=getattr(models.Distance, dist_name.upper()),
            ),
            "colbertv2.0": models.VectorParams(
                size=len(late_interaction_embeddings[0][0]),
                distance=getattr(models.Distance, dist_name.upper()),
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0)  #  Disable HNSW for reranking
            ),
        },
        sparse_vectors_config={
            sparse_retriever: models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )

    points = []
    for idx, (dense_embedding, bm25_embedding, late_interaction_embedding, doc) in enumerate(zip(dense_embeddings, bm25_embeddings, late_interaction_embeddings, documents)):

        point = models.PointStruct(
            id=idx,
            vector={
                "multilingual-e5-large": dense_embedding,
                "bm25": bm25_embedding.as_object(),
                "colbertv2.0": late_interaction_embedding,
            },
            payload={"document": doc}
        )
        points.append(point)

    operation_info = client.upsert(
        collection_name="hybrid-search",
        points=points
    )


    points = []

    for idx, doc in enumerate(documents):
        point = models.PointStruct(
            id=idx,
            vector={
                encoder: models.Document(text=doc, model=encoder),
                sparse_retriever: models.Document(text=doc, model="Qdrant/bm25"),
                "colbertv2.0": models.Document(text=doc, model="colbert-ir/colbertv2.0"),
            },
            payload=input_passages_dict[idx]
        )
        points.append(point)

    operation_info = client.upsert(
        collection_name="hybrid-search",
        points=points
    )

    return client, dense_embedding_model, bm25_embedding_model, late_interaction_embedding_model



def query_vector_db_once_qdrant(
    client, encoder, question:str, collection_name:str="dummy_name", k:int=5,
    dist_name:str="COSINE"):

    raw_answer = client.query_points(
        collection_name=collection_name,
        query=encoder.encode(question, normalize_embeddings=True).tolist(),
        limit=k).points

    processed_answer = {"user_question": question,
    "retrieved": [{
        "question": hit.payload["question"],
        "answer": hit.payload["answer"],
        dist_name.lower(): hit.score} for hit in raw_answer]
    }

    return processed_answer


def query_vector_db_list_qdrant(
    client, encoder, question_list:list[str],
    collection_name:str="dummy_name", k:int=5):
    answer_list = [
        query_vector_db_once_qdrant(
            client, encoder, q, collection_name, k) for q in question_list]

    return answer_list


def rag_setup_qdrant(
    config:dict[str], api_key_variable:str="OPENAI_API_KEY",
    qdrant_cloud_api_key_variable:str="QDRANT_CLOUD_API_KEY",
    from_scratch:bool=False):
    vector_db_client, encoder = setup_vector_db(
        encoder_name=config["encoder_name"],
        client_source=config["client_source"],
        qdrant_cloud_api_key=os.environ.get(qdrant_cloud_api_key_variable),
        from_scratch=from_scratch,
        input_folder=config["input_text_folder"],
        collection_name=config["collection_name"],
        dist_name=config["distance_type"],
        input_folder_qa=config["input_folder_qa"])

    api_client = OpenAI(api_key=os.environ.get(api_key_variable))

    print("RAG setup complete")
    return vector_db_client, encoder, api_client



def create_llm_qa_string(user_question:str, retrieved_docs:list[dict])->str:
    user_prompt = f"Answer the following question:\n<Question>{user_question}\n</Question>\n\n"
    user_prompt += "These are the documents passed to you by Senior Dietician Bot<Context>\n"

    for i, doc in enumerate(retrieved_docs):
        user_prompt += f"<Document{i+1}><Past_Question>{doc['question']}</Past_Question><Answer>{doc['answer']}</Answer></Document{i+1}>\n\n"

    user_prompt += "</Context>"
    # print(user_prompt)
    return user_prompt


def llm_call(client:OpenAI, user_prompt:str,
             system_propmt_path:str, model="gpt-4o-mini",
             temperature:float=0):

    with open(system_propmt_path, "r") as f:
        system_prompt = f.read()

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        model=model,
        temperature=temperature
    )

    return chat_completion.choices[0].message.content


def rag_query_once_qdrant(
    query:str, vector_db, encoder, api_client, config:dict[str])-> tuple[str, str, dict]:
    retrieved_doc_dict = query_vector_db_once_qdrant(
        vector_db, encoder, query,
        collection_name = config["collection_name"],
        k=config["retrieve_k"])

    print("Documents retrieved")
    # # Only return unique answers
    # unique_answers = []
    # [unique_answers.append(hit["answer"])
    #  for hit in retrieved_doc_dict["retrieved"]
    #  if hit["answer"] not in unique_answers]

    retrieved_docs_filtered = [
        doc for doc in retrieved_doc_dict["retrieved"]
        if doc["score"] >= config["min_similarity_threshold"]]

    user_prompt = create_llm_qa_string(query, retrieved_doc_dict["retrieved"])

    response = llm_call(
        client=api_client, user_prompt=user_prompt,
        system_propmt_path=config["llm_system_prompt_path"],
        model=config["llm_model"],
        temperature=config["llm_temperature"])

    return query, response, retrieved_doc_dict


def rag_query_list_qdrant(
    queries:list[str], vector_db, encoder, api_client, config:dict[str]) -> tuple[list[str],list[str],list[dict]]:
    """
    Iterative version of rag_query_once_qdrant() for evaluation
    """
    responses_raw = [rag_query_once_qdrant(
        q, vector_db, encoder, api_client, config) for q in queries]
    # summarize the list of payloads into two lists:
    #   - responses (str)
    #   - full_responses (dict with metadata)
    # (the list of queries was passed in as an input, doesn't need to be created)
    responses = [r[1] for r in responses_raw]
    retrieved_doc_dict_ls = [r[2] for r in responses_raw]

    return queries, responses, retrieved_doc_dict_ls



def main():
    try:
        import tomllib # type: ignore
    except ModuleNotFoundError:
        import tomli as tomllib

    with open("parameters.toml", mode="rb") as fp:
        config = tomllib.load(fp)

    parser=argparse.ArgumentParser(description="argument parser for rag-who")
    parser.add_argument("--config_name", nargs='?', default="default")
    args=parser.parse_args()
    print("Arguments parsed, parameters loaded")

    # print(args.config_name)
    # retrieve_and_eval(config_name=args.config_name)
    # vector_db_client, encoder, gen_api_client = rag_setup_qdrant(
    #     config_name=args.config_name, from_scratch=True)

    vector_db_client, dense_embedding_model, bm25_embedding_model, late_interaction_embedding_model = setup_vector_db(
        encoder_name=config[args.config_name]["encoder_name"],
        client_source=config[args.config_name]["client_source"],
        qdrant_cloud_api_key=os.environ.get("QDRANT_CLOUD_API_KEY"),
        from_scratch=True,
        input_folder=config[args.config_name]["input_text_folder"],
        collection_name=config[args.config_name]["collection_name"],
        dist_name=config[args.config_name]["distance_type"],
        input_folder_qa=config[args.config_name]["input_folder_qa"],
        sparse_retriever=config[args.config_name]["sparse_retriever"],
        hybrid=True)

    # vector_db_client, encoder, gen_api_client = rag_setup_qdrant(
    #     config=config[args.config_name], from_scratch=False)
    # print("#########################################")


    # query, response, full_response = rag_query_once_qdrant(
    #     "What should I snack on between lunch and dinner if I have diabetes?",
    #     vector_db_client, encoder, gen_api_client, config=config[args.config_name])
    # print(query, response, full_response, sep="\n")

    # print("#########################################")

    # # query, response, full_response = rag_query_once_qdrant(
    # #     "How often should a dog eat?",
    # #     vector_db_client, encoder, gen_api_client, config=config[args.config_name])
    # # print(query, response, full_response, sep="\n")


if __name__ == "__main__":
    main()
