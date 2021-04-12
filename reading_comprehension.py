import logging
logging.basicConfig(level=logging.INFO)
logging.info("Running reading comprehension module")

import torch
from transformers import BertForQuestionAnswering, BertTokenizer


# When you first run this module, it will take a few minutes to load the models.
# Model warm-up time is about 30 seconds - 1 minute, depending on your machine.
MODEL = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(MODEL)
# model = BertForQuestionAnswering.from_pretrained('./serving/models/')
tokenizer = BertTokenizer.from_pretrained(MODEL)


def get_model_predictions(question, answer_text):
    """
    TODO: improve this function!
    """
    input_ids = tokenizer.encode(question, answer_text)

    sep_index = input_ids.index(tokenizer.sep_token_id)

    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a

    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]), return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]
        
        else:
            answer += " " + tokens[i]

    all_data = {"answer": answer,
                "start_scores_max": start_scores.max(),
                "end_scores_max": end_scores.max(),
                "start_scores": start_scores,
                "end_scores": end_scores}

    return all_data
