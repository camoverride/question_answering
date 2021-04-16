"""
Test the QA model: test the actual BERT model, the tokenizer, and `get_model_predictions`
WARNING: Initialize the model server before running these tests.
TODO: warm up model server.
"""

import json
import unittest
import requests
from transformers import BertTokenizer
from reading_comprehension import get_model_predictions


# Model server config
URL = "localhost"
PORT = "8080"
MODEL_NAME = "bert_qa_squad"
MODEL_VERSION = "1"
MODEL_SERVER_ADDRESS = f"http://{URL}:{PORT}/v{MODEL_VERSION}/models/{MODEL_NAME}:predict"

# Test data along with tokenized forms and their ID's
QUERY = "What is the capital of France?"

CONTEXT = """
    The country's eighteen integral regions (five of which are situated overseas) span a combined area of 643,801 km2 (248,573 sq mi) and a total population of 67.4 million (as of March 2021).[12] France is a unitary semi-presidential republic with its capital in Paris, the country's largest city and main cultural and commercial centre. Other major urban areas include Lyon, Marseille, Toulouse, Bordeaux, Lille and Nice. France, including its overseas territories, has the most time zones of any country, with a total of twelve.
    """

INPUT_IDS = [101, 2054, 2003, 1996, 3007, 1997, 2605, 1029, 102, 1996, 2406, 1005, 1055, 7763, 9897, 4655, 1006, 2274, 1997, 2029, 2024, 4350, 6931, 1007, 8487, 1037, 4117, 2181, 1997, 4185, 2509, 1010, 3770, 2487, 2463, 2475, 1006, 24568, 1010, 5401, 2509, 5490, 2771, 1007, 1998, 1037, 2561, 2313, 1997, 6163, 1012, 1018, 2454, 1006, 2004, 1997, 2233, 25682, 1007, 1012, 1031, 2260, 1033, 2605, 2003, 1037, 22127, 4100, 1011, 4883, 3072, 2007, 2049, 3007, 1999, 3000, 1010, 1996, 2406, 1005, 1055, 2922, 2103, 1998, 2364, 3451, 1998, 3293, 2803, 1012, 2060, 2350, 3923, 2752, 2421, 10241, 1010, 16766, 1010, 17209, 1010, 16384, 1010, 22479, 1998, 3835, 1012, 2605, 1010, 2164, 2049, 6931, 6500, 1010, 2038, 1996, 2087, 2051, 10019, 1997, 2151, 2406, 1010, 2007, 1037, 2561, 1997, 4376, 1012, 102]

TOKENS = ['[CLS]', 'what', 'is', 'the', 'capital', 'of', 'france', '?', '[SEP]', 'the', 'country', "'", 's', 'eighteen', 'integral', 'regions', '(', 'five', 'of', 'which', 'are', 'situated', 'overseas', ')', 'span', 'a', 'combined', 'area', 'of', '64', '##3', ',', '80', '##1', 'km', '##2', '(', '248', ',', '57', '##3', 'sq', 'mi', ')', 'and', 'a', 'total', 'population', 'of', '67', '.', '4', 'million', '(', 'as', 'of', 'march', '2021', ')', '.', '[', '12', ']', 'france', 'is', 'a', 'unitary', 'semi', '-', 'presidential', 'republic', 'with', 'its', 'capital', 'in', 'paris', ',', 'the', 'country', "'", 's', 'largest', 'city', 'and', 'main', 'cultural', 'and', 'commercial', 'centre', '.', 'other', 'major', 'urban', 'areas', 'include', 'lyon', ',', 'marseille', ',', 'toulouse', ',', 'bordeaux', ',', 'lille', 'and', 'nice', '.', 'france', ',', 'including', 'its', 'overseas', 'territories', ',', 'has', 'the', 'most', 'time', 'zones', 'of', 'any', 'country', ',', 'with', 'a', 'total', 'of', 'twelve', '.', '[SEP]']

TOKEN_TYPE_IDS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Local tokenizer
tokenizer = BertTokenizer.from_pretrained("./models/tokenizer/")


class TestModel(unittest.TestCase):
    """
    Test both the model itself and the tokenizer.
    """
    def test_bert_qa_model(self):
        """
        Sends some real data to the BERT QA model to ensure that the response
        looks correct.
        """
        # Note: everything should be "attended to" (`attention_mask` is all 1's).
        input_ids = INPUT_IDS
        token_type_ids = TOKEN_TYPE_IDS
        attention_mask = [1 for _ in range(len(input_ids))]

        # Get the data into the format the BERT model will accept.
        signatures = [{"attention_mask": attention_mask,
                       "token_type_ids": token_type_ids, "input_ids": input_ids}]
        data = json.dumps({"signature_name": "serving_default", "instances": signatures})

        headers = {"content-type": "application/json"}
        json_response = requests.post(f"http://{URL}:{PORT}/v{MODEL_VERSION}/models/{MODEL_NAME}:predict", data=data, headers=headers)
        response_text = json.loads(json_response.text)

        # Make sure that predictions are returned.
        assert "predictions" in response_text

        for logit in ["start_logits", "end_logits"]:
            # Make sure that `start_logits and `end_logits` are returned.
            assert logit in response_text["predictions"][0]

            # Make sure that the data returned are all floats.
            assert all(isinstance(val, float) is True \
                    for val in response_text["predictions"][0][logit])

        # Make sure the data returned are equal length.
        assert len(response_text["predictions"][0]["start_logits"]) == \
               len(response_text["predictions"][0]["end_logits"])


    def test_bert_tokenizer(self):
        """
        Test the tokenizer on some real data.
        """
        # Get the input ids for the query + context, separated by "[SEP]"
        input_ids = tokenizer.encode(QUERY, CONTEXT)
        assert input_ids == INPUT_IDS

        # Make sure that these ids map to the correct words.
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        assert tokens == TOKENS

        # Make sure that the question is separeted from the context.
        sep_index = input_ids.index(tokenizer.sep_token_id)
        assert sep_index == 8


    def test_get_model_predictions(self):
        """
        Test both of these functions together in the `get_model_predictions` function!
        """
        model_output = get_model_predictions(QUERY, CONTEXT, MODEL_SERVER_ADDRESS)

        # Check that the answer is correct.
        assert model_output["answer"] == "paris"

        # Check that the start and end score max is a float.
        assert model_output["start_scores_max"].tolist() == 6.062509059906006
        assert model_output["end_scores_max"].tolist() == 7.04248046875

        # Check that the start and end score tensors are the right length.
        assert len(model_output["start_scores"]) == len(model_output["start_scores"]) == 130
