from scipy.spatial.distance import cosine
import numpy as np


def get_cosine_similarity(embedding_1, embedding_2) : 
    score = 1 - cosine(embedding_1, embedding_2)
    return score


def calculate_cosine_similarity(embeddings, query_embedding, related_papers) : 
    
    similarity_scores = []
    
    for i in range(len(related_papers)) : 
        score = get_cosine_similarity(embeddings[i], query_embedding)
        similarity_scores.append((score, f"{related_papers[i]}"))
        
    return similarity_scores