import os
from dotenv import load_dotenv
import google.generativeai as genai


def calculate_embeddings(papers): 
    
    embeddings = []
    
    for i in range(len(papers)): 
        paper_title = papers[i]
        
        paper_embedding = genai.embed_content(
            model = "models/text-embedding-004",
            content = f"{paper_title}",
            task_type = "similarity"
        )

        embeddings.append(paper_embedding["embedding"])

    return embeddings


def calculate_query_embedding(user_query): 
    
    query_embedding = genai.embed_content(
        model = "models/text-embedding-004",
        content = f"{user_query}",
        task_type = "similarity"
    )

    return query_embedding["embedding"]