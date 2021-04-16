"""
A demonstration of how to use this library.
Make sure you have a model server up and running!
"""

import logging
from answer_question import Answerer


# Model Server config
URL = "localhost"
PORT = "8080"
MODEL_NAME = "bert_qa_squad"
MODEL_VERSION = "1"
MODEL_SERVER = f"http://{URL}:{PORT}/v{MODEL_VERSION}/models/{MODEL_NAME}:predict"

# Build the query.
QUESTION = "what is the population of France?"
logging.info("Beginning QA")

answerer = Answerer(model_server_address=MODEL_SERVER)

logging.info(f"QUESTION: {QUESTION}")
ans = answerer.answer_question(QUESTION)
logging.info(f'ANSWER: {ans["answer"]["answer"]}')
