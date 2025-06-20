import os
import json
import pandas as pd



def process_text(path):
    with open(path, "r") as f:
        qa_data_ls = json.load(f)

    qa_data_df = pd.concat([pd.DataFrame(qa) for qa in qa_data_ls])

    # TODO: add more preprocessing and cleaning to account for less clean inputs

    return [{"question": q.strip().replace("’", "'"),
             "answer": a.strip().replace("’", "'")}
            for q,a in list(zip(qa_data_df.questions,qa_data_df.answer))]


def main():
    qa = process_text(
        os.path.join("data", "raw_input_files", "faq_data.json"))
    print(len(qa))
    print(qa[0])


if __name__ == "__main__":
    main()
