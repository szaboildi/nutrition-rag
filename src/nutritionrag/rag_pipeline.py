import glob
import os

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

from nutritionrag.preprocessing import process_text

import argparse
from openai import OpenAI

from uuid import uuid4



def setup_vector_db(
    encoder_name:str="intfloat/e5-base", client_source:str=":memory:",
    qdrant_cloud_api_key:str="None", force_replace_collection:bool=False,
    input_folder:str="data/raw_input_files",
    collection_name:str="dummy_name", dist_name:str="COSINE"):

    encoder = SentenceTransformer(encoder_name)
    if client_source == ":memory:":
        client = QdrantClient(client_source)
    else:
        client = QdrantClient(
            url=client_source, api_key=qdrant_cloud_api_key)

    if force_replace_collection:
        # Remove pre-existing collection
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)

        # Recreate collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
                distance=getattr(models.Distance, dist_name.upper()),
            ),
        )

        # Upload question-answer pairs from all files in the designated folder
        file_names = glob.glob(f"{input_folder}/*")
        for file_name in file_names:
            # Set up the passages
            input_passages_dict = process_text(path=file_name)

            print(f"Text cleaned in {file_name}")

            # Tokenize, embed and upload texts to Qdrant vector database
            client.upload_points(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid4()), vector=encoder.encode(
                            doc["question"], normalize_embeddings=True).tolist(),
                        payload=doc)
                    for doc in input_passages_dict],
            )

        print("Vector database created")
        return client, encoder

    # Otherwise load pre-existing collection and return it to the user
    else:
        print("Vector database loaded")
        return client, encoder


def query_vector_db_once_qdrant(
    client, encoder, question:str, config:dict[str]):

    raw_answer = client.query_points(
        collection_name=config["collection_name"],
        query=encoder.encode(question, normalize_embeddings=True).tolist(),
        limit=int(config["retrieve_k"])).points

    processed_answer = {"user_question": question,
    "retrieved": [{
        "question": hit.payload["question"],
        "answer": hit.payload["answer"],
        config["distance_type"].lower(): hit.score} for hit in raw_answer
                  if hit.score >= float(config["min_similarity_threshold"])]
    }

    return processed_answer


def query_vector_db_list_qdrant(
    client, encoder, question_list:list[str], config:dict[str]):
    answer_list = [
        query_vector_db_once_qdrant(
            client, encoder, q, config) for q in question_list]

    return answer_list


def rag_setup_qdrant(
    config:dict[str], api_key_variable:str="OPENAI_API_KEY",
    qdrant_cloud_api_key_variable:str="QDRANT_CLOUD_API_KEY"):
    vector_db_client, encoder = setup_vector_db(
        encoder_name=config["encoder_name"],
        client_source=config["client_source"],
        qdrant_cloud_api_key=os.environ.get(qdrant_cloud_api_key_variable),
        force_replace_collection=config["force_replace_collection"]=="True",
        input_folder=config["input_text_folder"],
        collection_name=config["collection_name"],
        dist_name=config["distance_type"])

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
        vector_db, encoder, query, config)

    print("Documents retrieved")
    # # Only return unique answers
    # unique_answers = []
    # [unique_answers.append(hit["answer"])
    #  for hit in retrieved_doc_dict["retrieved"]
    #  if hit["answer"] not in unique_answers]

    if len(retrieved_doc_dict["retrieved"]) > 0:
        user_prompt = create_llm_qa_string(query, retrieved_doc_dict["retrieved"])

        response = llm_call(
            client=api_client, user_prompt=user_prompt,
            system_propmt_path=config["llm_system_prompt_path"],
            model=config["llm_model"],
            temperature=config["llm_temperature"])
    else:
        response = "Sorry, I don't have information on that. Please try a different question."

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
    #     config_name=args.config_name)


    vector_db_client, encoder, gen_api_client = rag_setup_qdrant(
        config=config[args.config_name])
    print("#########################################")

    # answer = query_vector_db_once_qdrant(
    #     vector_db_client, encoder,
    #     "What should I snack on between lunch and dinner if I have diabetes?",
    #     config[args.config_name])
    # print(f'Retrieved {len(answer["retrieved"])} documents')

    query, response, retrieved_docs = rag_query_once_qdrant(
        "What should I snack on between lunch and dinner if I have diabetes?",
        vector_db_client, encoder, gen_api_client, config=config[args.config_name])
    print(query, response, retrieved_docs, sep="\n")

    print("#########################################")

    # query, response, full_response = rag_query_once_qdrant(
    #     "How often should a dog eat?",
    #     vector_db_client, encoder, gen_api_client, config=config[args.config_name])
    # print(query, response, full_response, sep="\n")


if __name__ == "__main__":
    main()
