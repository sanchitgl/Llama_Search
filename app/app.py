import streamlit as st
from backend import *
from beir.datasets.data_loader import GenericDataLoader
# Title of the app

# Add CSS styles for the containers
container_style = """
    <style>
        .container1 {
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 20px;
        }
        .container2 {
            /* Add styles for Container 2 if needed */
        }
    </style>
"""


def startup():
    model, retriever = load_distilbert_bm25()
    dataset = "scifact"
    dataset_path = f"../datasets/{dataset}"
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
    return model, retriever, corpus, queries, qrels


if 'startup' not in st.session_state:
    st.session_state.startup = True
    st.session_state.model, st.session_state.retriever, st.session_state.corpus, st.session_state.queries, st.session_state.qrels = startup()


def get_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None 

st.title("SADA Search Engine")

# Dropdown for selecting a dummy query
query_options = list(st.session_state.queries.values())[:10]

query = st.selectbox("Choose a Query:", query_options)

qid = get_key(st.session_state.queries, query)



# Dropdown for selecting a model
model_options = [
    "BM25",
    "DistilBert",
    "DistilBert + BM25"
]
model = st.selectbox("Choose a model:", model_options)


# Placeholder for the results
results_placeholder = st.empty()

# Sidebar for accuracy and metrics
sidebar = st.sidebar
sidebar.header("Metrics")
accuracy_sidebar = sidebar.empty()
dcg_sidebar = sidebar.empty()
map_sidebar = sidebar.empty()
recall_sidebar = sidebar.empty()
precision_sidebar = sidebar.empty()

# Button to submit the query
if st.button("Submit"):
    if query and model:
        # Mockup: Replace this part with your model's logic to get the results and accuracy

        if model == "BM25":
            print("Running bm25 ...")
            results = bm25_get_result(qid)
        elif model == "DistilBert":
            print("Running DistilBert ...")
            results = get_distilbert_bm25_result(qid, st.session_state.retriever)
        else:
            print("Running DistilBert+BM25 ...")
            results = ensemble_distilbert_bm25(qid, st.session_state.retriever)

        ndcg, _map, recall, precision = st.session_state.retriever.evaluate(st.session_state.qrels, results, st.session_state.retriever.k_values)

        # For demonstration, we're using hardcoded results and metrics
        top_10_results = [k for k,v in results[qid].items()][:10]

        # ndcg, _map, recall, precision

        dcg_10 = ndcg['NDCG@10']
        _map_10 = _map['MAP@10'] 
        recall_10 = recall['Recall@10'] 

        # Displaying the top 10 results
        results_placeholder.subheader("Top 10 Results:")
        display_rows = [st.columns(1) for _ in range(len(top_10_results))]
            
        for idx, result in enumerate(top_10_results):
            # st.write(st.session_state.corpus[result])
            
            # selecting container and adding text to it
            col = display_rows[idx]
            tile = col[0].container(height=200)
            # tile.caption(f"ID:{st.session_state.corpus[result]['title'].strip()} - Date:{datetime.today().strftime('%Y-%m-%d')} ")
            tile.markdown(f"**{st.session_state.corpus[result]['title'].strip()}**")
            tile.markdown(f"*{st.session_state.corpus[result]['text'].strip()}*")

        dcg_sidebar.write(f"**nDCG@10:** {dcg_10}")
        map_sidebar.write(f"**MAP@10:** {_map_10}")
        recall_sidebar.write(f"**Recall@10:** {recall_10}%")
    else:
        st.write("Please select a query and a model.")
