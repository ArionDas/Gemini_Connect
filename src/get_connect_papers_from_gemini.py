import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
import json
import re
import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

from gemini_api_configure import configure_GEMINI_API
from gemini_model_variant import choose_GEMINI_model_variant
from gemini_cache import cache_GEMINI_model
from gemini_model_setup import GEMINI_model_setup
from prepare_prompt import prepare_USER_prompt
from get_model_response import get_GEMINI_response
from clean_responses import extract_related_papers
from dataset import get_title_names, get_summaries
from calculate_embeddings import calculate_embeddings, calculate_query_embedding
from cosine_similarity import calculate_cosine_similarity


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def create_connected_papers_plot(related_papers, similarity_scores):

    # Data: Similarity scores and paper titles
    data = similarity_scores

    # Step 1: Calculate cos-inverse (arccos) of similarity scores
    angles = [np.arccos(score[0]) for score in data]  # Angles in radians

    # Step 2: Add 20 degrees to each angle and convert to degrees
    angles = [(np.degrees(angle)) % 360 for angle in angles]  # Add 20 degrees, ensure within [0, 360]

    # Step 3: Ensure a minimum of 5 degrees between each angle
    angles.sort()  # Sort angles for sequential adjustment
    min_diff = 5  # Minimum difference in degrees

    for i in range(1, len(angles)):
        if angles[i] - angles[i - 1] < min_diff:
            angles[i] = angles[i - 1] + min_diff

    # Wrap around angles > 360
    angles = [angle % 360 for angle in angles]

    # Convert back to radians for polar plotting
    angles = [np.radians(angle) for angle in angles]

    titles = [item[1] for item in data]

    # Step 4: Define unique colors for each paper
    colormap = get_cmap('tab10')  # Use a colormap (e.g., tab10)
    colors = [colormap(i % 10) for i in range(len(data))]

    # Step 5: Plotting the papers
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})

    # Scatter plot with unique colors
    for i, (angle, title) in enumerate(zip(angles, titles)):
        ax.scatter(angle, 1, c=[colors[i]], s=150, label=title)  # Fixed radius at 1

    # Annotate papers with dynamic positioning
    for i, (angle, title) in enumerate(zip(angles, titles)):
        text_angle = np.degrees(angle)
        x_offset = 1.1  # Slightly outside the point radius
        ax.text(
            angle, x_offset, title, color=colors[i], fontsize=8,
            ha='center', va='center', rotation=text_angle, rotation_mode='anchor'
        )

    # Style the plot
    ax.set_yticks([])  # Remove radial grid lines
    ax.set_xticks([])  # Remove angle grid lines

    # Display the plot
    plt.tight_layout()
    plt.show()



def get_connected_papers_from_GEMINI(user_query : str, model_number : int, cache_time : int):
    
    configure_GEMINI_API()
    
    model = choose_GEMINI_model_variant(model_number)
    
    prompt = prepare_USER_prompt(user_query)
    related_papers = get_GEMINI_response(user_query, model_number, cache_time, model, prompt)
    
    embeddings = calculate_embeddings(related_papers)
    query_embeding = calculate_query_embedding(user_query)
    
    similarity_scores = calculate_cosine_similarity(embeddings, query_embeding, related_papers)
    
    create_connected_papers_plot(related_papers, similarity_scores)
    
    return related_papers