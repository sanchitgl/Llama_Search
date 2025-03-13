import os
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

# Set CUDA devices BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"]= "3"

tokenizer_max_len = 512

dataset = "msmarco_tiny"

dataset_path = "../beir/datasets/msmarco_tiny/"
corpus_file = "tiny_collection.json"
queries_file = "topics.dl20.txt"
qrels_test_file = "qrels.dl20-passage.txt"
training_set = "msmarco_triples.train.tiny.tsv"


import collections
import pytrec_eval
import json

def load_queries(path):
    """Returns a dictionary whose keys are query ids and values are query texts."""
    queries = {}
    with open(path) as f:
        for line in f:
            query_id, query_text = line.strip().split('\t')
            queries[query_id] = query_text
    return queries


def load_corpus_tsv(path):
    """Returns a dictionary whose keys are passage ids and values are passage texts."""
    corpus = {}
    with open(path) as f:
        for line in f:
            passage_id, passage_text = line.strip().split('\t')
            corpus[passage_id] = {'text': passage_text}
    return corpus

def load_corpus_json(path):
    with open(path, 'r') as corpus_f:
        corpus_json = json.load(corpus_f)
    return corpus_json


def load_qrels(path):
    with open(path, 'r') as f_qrel:
        qrels = pytrec_eval.parse_qrel(f_qrel)

    return qrels


def load_triplets(path):
    triplets = []
    with open(path) as f:
        for line in f:
            query, positive_passage, negative_passage = line.strip().split('\t')
            triplets.append([query, positive_passage, negative_passage])
    return triplets

# Don't need to load triplet for training bm25
# triplets = load_triplets('msmarco_triples.train.tiny.tsv')

qrels = load_qrels(f"{dataset_path}{qrels_test_file}")
queries = load_queries(f"{dataset_path}{queries_file}")
print("Loading corpus into memory ...")
corpus = load_corpus_json(f"{dataset_path}{corpus_file}")



import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = get_model('castorini/repllama-v1-7b-lora-passage')

model = model.to(device) # Moving model to GPU

def get_embed(input):

    tokenized_inputs = tokenizer(f'{input}</s>', return_tensors='pt')
    tokenized_inputs = tokenized_inputs.to(device)
    with torch.no_grad():
        # compute query embedding
        outputs = model(**tokenized_inputs)
        embedding = outputs.last_hidden_state[0][-1] #outputs.last_hidden_state[:,-1,:]  # Get embedding of last token i.e. </s>
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return embedding

from tqdm import tqdm

query_embeddings = {}
doc_embeddings = {}

print("Encoding queries ...")
for k,q in tqdm(queries.items()):
    query_embed = get_embed(q)
    query_embeddings[k] = query_embed


import pickle

# Can't save tensors to json
with open(f"{dataset_path}queries_llamaEmbed.pickle", 'wb') as f:
    pickle.dump(query_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)


print("Encoding passages ...")
i = 0
for k,q in tqdm(corpus.items()):
    i+=1
    doc_embed = get_embed(q['text'])
    doc_embeddings[k] = doc_embed

    if i%50_000==0:
        with open(f"{dataset_path}tiny_collection_llamaEmbed_{i}.pickle", 'wb') as f:
            pickle.dump(doc_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open(f"{dataset_path}tiny_collection_llamaEmbed.json", 'w') as f:
#     json.dump(doc_embeddings, f)

with open(f"{dataset_path}tiny_collection_llamaEmbed.pickle", 'wb') as f:
    pickle.dump(doc_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

