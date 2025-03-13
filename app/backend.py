
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from datetime import datetime
import os
import pickle

dataset = "scifact"
dataset_path = f"../datasets/{dataset}"

corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")


def load_distilbert(model_path):
    print("Loading model ...")
    #################################
    # Loading DistilBERT
    #################################
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval import models
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

    ## Load retriever from saved model
    model = DRES(models.SentenceBERT(model_path), batch_size=256)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    print("Model loaded ...")
    return model, retriever


def load_distilbert_random():
    
    model_name = "distilbert-base-uncased" 
    model_save_path = os.path.join("../", "output", "{}-v1-{}".format(model_name, dataset)) 
    model, retriever = load_distilbert(model_save_path)
    return model, retriever


def load_distilbert_bm25():
    model_name = "distilbert-base-uncased" 
    model_save_path = os.path.join(os.getcwd(), "../output", "{}-v2-{}-bm25-hard-negs".format(model_name, dataset))
    model, retriever = load_distilbert(model_save_path)
    return model, retriever


def bm25_get_result(qid):
    score_path = "../datasets/scifact/scifact_bm25_scores.pickle"

    with open(score_path, 'rb') as f:
        results_bm25 = pickle.load(f)
    
    docids = results_bm25[qid]
    return {qid: docids}


def get_distilbert_r_result(qid, retriever):
    new_query = {qid: queries[qid], '0': ""}
    print(new_query)
    dense_result = retriever.retrieve(corpus, new_query)
    return dense_result


def get_distilbert_bm25_result(qid, retriever):
    new_query = {qid: queries[qid], '0': ""}
    dense_result = retriever.retrieve(corpus, new_query)
    return dense_result



def get_maxmin(results):
    max_score = -1
    min_score = 999999
    for q_id, q in results.items():
        for doc_id, score in q.items():
            max_score = max(score, max_score)
            min_score = min(score, min_score)

    return min_score, max_score

def ensemble_score(x,y):
    mu = 0.5
    return mu*x + (1-mu)*y

def normalize_results(results, min_score, max_score):
    for q_id, q in results.items():
        for doc_id, score in q.items():
            results[q_id][doc_id] = (score-min_score)/(max_score-min_score)

    return results

def ensemble_distilbert_bm25(qid, retriever):
    #### Retrieve dense results (format of results is identical to qrels)
    new_query = {qid: queries[qid], '0': ""}
    dense_result = retriever.retrieve(corpus, new_query)
    bm25_result = bm25_get_result(qid)

    # Get range to normalize both
    min_distilbert_score, max_distilbert_score = get_maxmin(dense_result)
    min_bm25_score, max_bm25_score = get_maxmin(bm25_result)

    dense_result = normalize_results(dense_result, min_distilbert_score, max_distilbert_score)
    bm25_result = normalize_results(bm25_result, min_bm25_score, max_bm25_score)

    combined_result = {}

    for q_id_1, q_1 in dense_result.items():
        combined_result[q_id_1] = {}
        for doc_id_1, score_1 in q_1.items():
            
            score_2 = 0
            if q_id_1 in bm25_result and bm25_result[q_id_1].get(doc_id_1,None)!=None:
                score_2 = bm25_result[q_id_1][doc_id_1]
                del bm25_result[q_id_1][doc_id_1] # So that same query-doc pair is not added to combined result twice
            
            combined_score = ensemble_score(score_1, score_2)
            combined_result[q_id_1][doc_id_1] = combined_score


    # Now add remaining bm25 results in combined dict
    for q_id_2, q_2 in bm25_result.items():
        for doc_id_2, score_2 in q_2.items():
            score_1 = 0
            combined_score = ensemble_score(score_1, score_2)
            combined_result[q_id_1][doc_id_1] = combined_score
            
    return combined_result

