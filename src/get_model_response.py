import os
from dotenv import load_dotenv
import google.generativeai as genai

from dataset import get_title_names, get_summaries
from gemini_model_variant import choose_GEMINI_model_variant
from gemini_api_configure import configure_GEMINI_API
from gemini_cache import cache_GEMINI_model
from gemini_model_setup import GEMINI_model_setup, safety_settings, model_hyperparameters
from prepare_prompt import prepare_USER_prompt
from clean_responses import extract_related_papers



def get_GEMINI_response(user_query : str, cache_time : int, model, prompt):
    
    cache = cache_GEMINI_model(cache_time)
    
    prompt = prepare_USER_prompt(user_query)
    model = genai.GenerativeModel.from_cached_content(
        cached_content=cache, 
        generation_config=model_hyperparameters(None, 0, 0.0), 
        safety_settings=safety_settings()
    )
    
    try:
        response = model.generate_content(prompt)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    related_papers = extract_related_papers(response.text)[:20]
    
    return related_papers


