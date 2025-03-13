import os
os.environ["CUDA_VISIBLE_DEVICES"]= "6"


import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

logging.set_verbosity_info()


model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-IR2"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25
max_seq_length = None
packing = False
device_map = {"": 0}


import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig


import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map={"": 0})
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf')
model = get_model("../models/Llama_new")

model = model.to(device)


import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
import logging
import collections
import pytrec_eval

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

# Define file paths
corpus_file = "tiny_collection.json"
queries_file = "topics.dl20.txt"
qrels_test_file = "qrels.dl20-passage.txt"
dataset_path = "../beir/datasets/msmarco_tiny/"

# Load queries
def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            query_id, query_text = line.strip().split('\t')
            queries[query_id] = query_text
    return queries

# Load qrels
def load_qrels(path):
    with open(path, 'r') as f_qrel:
        qrels = pytrec_eval.parse_qrel(f_qrel)
    return qrels

# Load corpus
def load_corpus_json(path):
    with open(path, 'r') as corpus_f:
        corpus_json = json.load(corpus_f)
    return corpus_json

qrels = load_qrels(os.path.join(dataset_path, qrels_test_file))
queries = load_queries(os.path.join(dataset_path, queries_file))
corpus = load_corpus_json(os.path.join(dataset_path, corpus_file))



def get_embed_dataset(input_lst):

    input_txt = [f'{input}</s>' for input in input_lst['text']]

    tokenized_inputs = tokenizer(input_txt, return_tensors='pt', padding="max_length", truncation=True, max_length=tokenizer_max_len)
    tokenized_inputs = tokenized_inputs.to(device)
    with torch.no_grad():
        # compute query embedding
        outputs = model(**tokenized_inputs)
        embedding = outputs.last_hidden_state[:,-1,:]   #outputs.last_hidden_state[0][-1] # Get embedding of last token i.e. <s>
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return embedding

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
with open(f"{dataset_path}queries_newllamaEmbed.pickle", 'wb') as f:
    pickle.dump(query_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Encoding passages ...")
i = 0
for k,q in tqdm(corpus.items()):
    i+=1
    doc_embed = get_embed(q['text'])
    doc_embeddings[k] = doc_embed

    if i%50_000==0:
        with open(f"{dataset_path}corpus_newllamaEmbed_{i}.pickle", 'wb') as f:
            pickle.dump(doc_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)


with open(f"{dataset_path}corpus_newllamaEmbed.pickle", 'wb') as f:
    pickle.dump(doc_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

