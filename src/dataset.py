import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd


def get_title_names():
    
    arxiv_dataset = pd.read_csv("data/arxiv-metadata-oai-snapshot.csv")
    titles = []

    for i in range(len(arxiv_dataset)): 
        title = arxiv_dataset["title"][i]
        titles.append(title)
    
    return titles


def get_summaries():
    
    arxiv_dataset = pd.read_csv("data/arxiv-metadata-oai-snapshot.csv")
    
    ### Reshuffling the dataset as Gemini can't take in all the summary contexts at once, so we'll passing a sample
    arxiv_dataset = arxiv_dataset.sample(frac=1).reset_index(drop=True)
    
    summaries = []

    for i in range(len(arxiv_dataset)): 
        title = arxiv_dataset["title"][i]
        summary = arxiv_dataset["summary"][i]
        json_summary = {f"{title}" : f"{summary}"}
        summaries.append(json_summary)
        
    return summaries