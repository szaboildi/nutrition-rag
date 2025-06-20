import os
# os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"  # Disables problematic inspection

import streamlit as st
from nutritionrag.rag_pipeline import rag_setup_qdrant, rag_query_once_qdrant
# Config
try:
    import tomllib # type: ignore
except ModuleNotFoundError:
    import tomli as tomllib

with open(os.path.join("parameters.toml"), mode="rb") as fp:
    config = tomllib.load(fp)

config_name = "default"
from_scratch = False


# Setup
vector_db_client, encoder, llm_client = rag_setup_qdrant(
    config=config[config_name], from_scratch=from_scratch)



# App functionality
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.placeholder = "Your query goes here"

query = st.text_input(
    "Please put your query below",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder=st.session_state.placeholder)


if query:
    st.empty()
    rag_response = rag_query_once_qdrant(
        query, vector_db_client, encoder, llm_client, config[config_name])
    st.subheader(rag_response[1])

    if len(rag_response[2]["retrieved"])==0:
        st.write("We were unable to find what you're looking for")
    else:
        st.markdown("---")
        st.write("We based this response on the following answers from our catalogue:")
        st.markdown("---")
        unique_answers = []
        for item in rag_response[2]["retrieved"]:
            if item["answer"] not in unique_answers:
                unique_answers.append(item["answer"])
                st.write(f'Question: {item["question"]}')
                st.write(f'Answer: {item["answer"]}')
                st.write(f'Similarity score ({config[config_name]["distance_type"].lower()}): {item[config[config_name]["distance_type"].lower()]}')
                st.markdown("---")
            if len(unique_answers) > 2:
                break
