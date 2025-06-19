import os
import json
import pandas as pd



def process_text(
    path, force_write_qa_passages:bool=False, input_folder_qa:str="data",
    relevance_score_file_prefix:str="sample_qa_passage_lvl",
    sample_qa_file:str="sample_qa.json"):
    with open(path, "r") as f:
        qa_data_ls = json.load(f)

    qa_data_df = pd.concat([pd.DataFrame(qa) for qa in qa_data_ls])

    # TODO: add more preprocessing and cleaning to account for less clean inputs

    return [{"question": q, "answer": a}
            for q,a in list(zip(qa_data_df.questions,qa_data_df.answer))]


def main():
    qa = process_text(
        os.path.join("data", "raw_input_files", "faq_data.json"))
    print(len(qa))
    print(qa[0])


if __name__ == "__main__":
    main()
