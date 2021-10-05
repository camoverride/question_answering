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
logging.info("Beginning QA")

answerer = Answerer(model_server_address=MODEL_SERVER)

print("Ask a question, i.e. 'what is the capital of France?'")
print("To exit the session, type 'end'")

# Event loop where questions can be asked
question = "A"
while question not in ["end", "End", "stop", "Stop"]:

    question = input("QUESTION: ")
    ans = answerer.answer_question(question)
    print("ANSWER:")
    print(f'{ans["answer"]["answer"]}')
