# Import libraries
import streamlit as st
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from datetime import datetime
import os
import pickle

dataset = "scifact"
dataset_path = f"../datasets/{dataset}"
show_txt_length = 400

corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")

queries_to_select = {}
for i, qid in enumerate(queries):
    queries_to_select[qid] = queries[qid] 
    if i>4:
        break

# corpus to df
doc_ids = []
docs = []
for docid, doc in corpus.items():
    doc_ids.append(docid)
    docs.append(doc['text'])

corpus_df = pd.DataFrame({'docid': doc_ids, 'docs': docs})
# corpus_df.head()



# Use a text_input to get the keywords to filter the dataframe
query_option = st.selectbox("Select a query", tuple(queries_to_select.values()))
model_option = st.selectbox("Select a method", ("BM25","DistilBERT-Random Trained","DistilBERT-BM25 Trained","DistilBERT-Random + BM25 Ensemble","DistilBERT-BM25 + BM25 Ensemble"))

# Show the results, if you have a text_search
if query_option and model_option:
    # st.write(corpus_df[:6])
    st.title("Results")
    
    df_search = corpus_df[:10]
    display_rows = [st.columns(1) for _ in range(len(df_search))]

    i = 0
    for col in display_rows:
        print(col[0])
        tile = col[0].container(height=200)
        tile.caption(f"ID:{df_search.iloc[i]['docid'].strip()} - Date:{datetime.today().strftime('%Y-%m-%d')} ")
        tile.markdown(f"*{df_search.iloc[i]['docs'][:show_txt_length].strip()}* ...")
        i += 1

    # for n_row, row in df_search.reset_index().iterrows():
    #     container = st.container(border=True)
    #     container.write("This is inside the container")


    # # Another way to show the filtered results
    # # Show the cards
    # N_cards_per_row = 4
    # if text_search:
    #     for n_row, row in df_search.reset_index().iterrows():
    #         i = n_row%N_cards_per_row
    #         if i==0:
    #             st.write("---")
    #             cols = st.columns(N_cards_per_row, gap="small") #large
    #         # draw the card
    #         with cols[n_row%N_cards_per_row]:
    #             st.caption(f"ID:{row['docid'].strip()} - Date:{datetime.today().strftime('%Y-%m-%d')} ")
    #             st.markdown(f"*{row['docs'][:show_txt_length].strip()}* ...")
                