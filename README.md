# Question answering

This project implements a question answering system using Wikipedia as a resource. The system that performs document retrieval is Wikipedia's own [elastic search](https://en.wikipedia.org/wiki/Elasticsearch) engine. The model that does reading comprehension is [BERT](https://arxiv.org/abs/1810.04805) fine-tuned on Wikipedia, courtesy of the [transformers](https://huggingface.co/transformers/) library 🥰


## Run

Install the requirements:

- `pip install -r requirements-dev.txt`

Download the model as a docker servable and boot it up:

- `docker pull camoverride/bert-squad-qa-large:v0.1`
- `docker run -t --rm -p 8080:8080 camoverride/bert-squad-qa-large:v0.1`

Or build it locally and then boot it up:

- `docker build -t camoverride/bert-squad-qa-large:v0.1 .`
- `docker run -t --rm -p 8080:8080 camoverride/bert-squad-qa-large:v0.1`

Test it out:

- `python -m unittest tests/test_bert_model.py`


## Under the hood

Question answering systems all operate in two steps:

- find some relevant documents. This process is called _document retrieval_ and the software that does this is called a _search engine_. This is implemented in `document_retrieval.py`.
- read through the documents and find the answer. This process is called _reading comprehension_ and is performed by a model like BERT. This is implemented in `reading_comprehension.py`

Additionally, there are a number of odds and ends that need to be tied up. This is all done in the `answer_question.py` module:

- chunking the retrieved documents into pieces that are small enough to be evaluated by the model.
- deciding which of the model's evaluations contains the correct answer.
