"""
Downloads the huggingface model and saves it in a format that can
be used by tfx (tensorflow serving)
"""

import torch
from transformers import TFBertForQuestionAnswering, BertTokenizer


MODEL = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = TFBertForQuestionAnswering.from_pretrained(MODEL)
tokenizer = BertTokenizer.from_pretrained(MODEL)

model.save_pretrained(save_directory="models", saved_model=True, version=1)
