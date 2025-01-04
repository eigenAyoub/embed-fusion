from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel


## Plan:


## load wiki:





## model IDS 
bge_ID = 'BAAI/bge-small-en-v1.5'
arctic_ID = "Snowflake/snowflake-arctic-embed-m-v1.5"

bge_tokenizer = AutoTokenizer.from_pretrained(bge_ID)
bge = AutoModel.from_pretrained(bge_ID)
bge.eval()

arctic_tokenizer = AutoTokenizer.from_pretrained(arctic_ID)
arctic = AutoModel.from_pretrained(arctic_ID, add_pooling_layer=False)
arctic.eval()

# some e.g.,
queries  = ['what is snowflake?', 'Where can I get the best tacos?']
documents = ['The Data Cloud!', 'Mexico City of Course!']

documents = ['The Data Cloud! daTA ddkjkj fuck off piece of shit', 'Mexico City of Course!']
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '

bge_encoded_input = bge_tokenizer(documents, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    bge_output = bge(**bge_encoded_input)
    sentence_embeddings = bge_output[0][:, 0]

# normalize embeddings
sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)

# Add query prefix and tokenize queries and docs.
queries_with_prefix = [f"{QUERY_PREFIX}{q}" for q in queries]
query_tokens = arctic_tokenizer(queries_with_prefix, padding=True, truncation=True, return_tensors='pt', max_length=512)
document_tokens =  arctic_tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512)

# Use the arctic to generate text embeddings.
with torch.inference_mode():
    query_embeddings = arctic(**query_tokens)[0][:, 0]
    document_embeddings = arctic(**document_tokens)[0][:, 0]

# Remember to normalize embeddings.
query_embeddings = normalize(query_embeddings)
document_embeddings = normalize(document_embeddings)

# Scores via dotproduct.
scores = query_embeddings @ document_embeddings.T


