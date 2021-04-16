"""
A demonstration of how to use this library.
Make sure you have a model server up and running!
"""

import logging
import yaml
from answer_question import Answerer


# Change this to debug if you want to see documents being downloaded.
logging.basicConfig(level=logging.INFO)

# Model Server config
with open("model_server_config.yaml") as conf:
    config = yaml.load(conf, Loader=yaml.FullLoader)
    MODEL_SERVER = f"http://{config['model_server_url']}:{config['model_server_port']}/v{config['model_version']}/models/{config['model_name']}:predict"

# Build the query.
QUESTION = "what is the population of France?"
logging.info("Beginning QA")

answerer = Answerer(model_server_address=MODEL_SERVER)

print(f"QUESTION: {QUESTION}")
ans = answerer.answer_question(QUESTION)
print(f'ANSWER: {ans["answer"]["answer"]}')
