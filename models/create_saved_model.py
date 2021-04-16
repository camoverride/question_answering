"""
Downloads the huggingface model and saves it in a format that can
be used by tfx (tensorflow serving). Package this up as a Docker
image and you can run it locally.
"""

import torch
from transformers import TFBertForQuestionAnswering


model = TFBertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

model.save_pretrained(save_directory="models/bert_qa_squad", saved_model=True, version=1)
