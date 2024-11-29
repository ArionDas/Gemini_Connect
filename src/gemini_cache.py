import os
import google.generativeai as genai
from google.generativeai import caching
import datetime
import time


from dataset import get_title_names



def cache_GEMINI_model(minutes : int):
    
    title_names = get_title_names()
    
    cache = caching.CachedContent.create(
                model="gemini-1.5-flash-001",
                display_name="connected papers",  # used to identify the cache
                system_instruction=(
                    'You are an expert at Generative AI Research, LLMs and machine learning.'
                    "Your task is to come up with 20 most relevant and related research papers to the given user's paper."
                    "Please don't provide any explanation or any unwanted text, just provide the 20 title names as text."
                ),
                contents=[title_names],
                ttl=datetime.timedelta(minutes=60)
            )
    
    return cache