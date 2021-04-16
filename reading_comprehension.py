"""
This module contains the function `get_model_predictions` which calls an external
model server to perform reading comprehension.
"""

import json
import logging
import torch
import requests
from transformers import BertTokenizer


logging.basicConfig(level=logging.INFO)
logging.info("Running reading comprehension module")


tokenizer = BertTokenizer.from_pretrained("./models/tokenizer/") # load locally


def get_model_predictions(question: str, answer_text: str, model_server_address: str) -> dict:
    """
    This function accepts a question and some text that contains the answer and
    returns a dict containing the answer along with the max scores for the start
    and end indexes, along with the start/end scores for every index.

    Parameters
    question : str
        A question or query like "what is the capital of France?"

    answer : str
        Some context that contains the answer, like "Paris is the capital of France..."

    Returns
    -------
    dict
        A dict with the keys `answer`, `start_scores_max`, `end_scores_max`, `start_scores`,
        and `end_scores`. The latter two are vectors across the entire tokenized `answer_text`.
    """
    # Encode the question and answer as intergers.
    input_ids = tokenizer.encode(question, answer_text)

    # Get the special "[SEP]" token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # Build the vector of 0's and 1's that distinguishes the question from the answer.
    len_question = sep_index + 1
    len_answer = len(input_ids) - len_question
    token_type_ids = [0] * len_question + [1] * len_answer

    # Add the attention mask, which is all 1's (attend to everything)
    attention_mask = [1 for _ in range(len(input_ids))]

    # Build the request that is sent to the server.
    signatures = [{"attention_mask": attention_mask,
                "token_type_ids": token_type_ids, "input_ids": input_ids}]
    data = json.dumps({"signature_name": "serving_default", "instances": signatures})

    headers = {"content-type": "application/json"}
    # json_response = requests.post(f"http://{URL}:{PORT}/v{MODEL_VERSION}/models/{MODEL_NAME}:predict", data=data, headers=headers)
    json_response = requests.post(model_server_address, data=data, headers=headers)

    response_text = json.loads(json_response.text)

    # Get the start and end scores from the response.
    start_scores = torch.tensor(response_text["predictions"][0]["start_logits"])
    end_scores = torch.tensor(response_text["predictions"][0]["end_logits"])

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Convert back to tokens so that the answer can be a string.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]
        else:
            answer += " " + tokens[i]

    # Set up a dict to organize the data returned by the model.
    all_data = {"answer": answer,
                "start_scores_max": start_scores.max(),
                "end_scores_max": end_scores.max(),
                "start_scores": start_scores,
                "end_scores": end_scores}

    return all_data
