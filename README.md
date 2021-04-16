# Question answering

This project implements a question answering system using Wikipedia as a resource. The system that performs document retrieval is Wikipedia's own [elastic search](https://en.wikipedia.org/wiki/Elasticsearch) engine. The model that does reading comprehension is [BERT](https://arxiv.org/abs/1810.04805) fine-tuned on Wikipedia, courtesy of the [transformers](https://huggingface.co/transformers/) library 🥰

~~~python
from answer_question import Answerer

question = "what is the population of France?"

answerer = Answerer(model_server_address="http://localhost:8080/v1/models/bert_qa_squad:predict")

ans = answerer.answer_question(question)["answer"]["answer"] # 67 . 4 million
~~~

To learn a little more about this, check out [the blog post](https://camtsmith.com/articles/2021-04/question-answering-with-bert).


## Run

Install the requirements:

- `pip install -r requirements-dev.txt`

Download the model as a docker servable and boot it up:

- `docker pull camoverride/bert-squad-qa-large:v0.2`
- `docker run -t --rm -p 8080:8080 camoverride/bert-squad-qa-large:v0.2`

Or build it locally and then boot it up:

- `python models/create_saved_model.py` (creates the folder `models/bert_qa_squad` because the model is too large to store in git)
- `docker build -t camoverride/bert-squad-qa-large:v0.2 .`
- `docker run -t --rm -p 8080:8080 camoverride/bert-squad-qa-large:v0.2`

Test it out:

- `python demo.py`

If you want a closer look at how the functions all work, run the tests:

- `python -m unittest tests/test_bert_model.py`
- `python -m unittest tests/test_document_retrieval.py`

If you need to tweak the model server configuration, check out `model_server_config.yaml`.


## Under the hood

Question answering systems all operate in two steps:

- Find some relevant documents. This process is called _document retrieval_ and the software that does this is called a _search engine_. This is implemented in `document_retrieval.py`. This function requires an active internet connection.
- Read through the documents and find the answer. This process is called _reading comprehension_ and is performed by a model like BERT. This is implemented in `reading_comprehension.py`. This function requires a model server.

These two steps (plus some post-processing) are implemented in the `Answerer` class, which lives in `answer_question.py`.

I built the model server from a `SavedModel` that I run with tensorflow serving. However, it was too big to save in this repo. A module that re-creates this model artifact is in `models/create_saved_model.py`.
