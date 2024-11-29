import os
from dotenv import load_dotenv
import google.generativeai as genai


from gemini_model_variant import choose_GEMINI_model_variant
from gemini_api_configure import configure_GEMINI_API
from gemini_cache import cache_GEMINI_model



def safety_settings():
    
    ## We don't want to see any harmful content
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_ONLY_HIGH",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH",
        },
    ]
    
    return safety_settings



def model_hyperparameters(stop_sequences, temperature : int, top_p : int):
    
    generation_config = genai.GenerationConfig(
        stop_sequences = stop_sequences,
        temperature = temperature,
        top_p = top_p,
    )
    
    return generation_config



def GEMINI_model_setup(model_variant : int, cache_time : int):
     
    safety_settings = safety_settings()
    generation_config = model_hyperparameters(None, 0, 0.0)
    
    configure_GEMINI_API()
    
    cached_content = cache_GEMINI_model(cache_time)
    
    model = choose_GEMINI_model_variant(model_variant)
    
    model = genai.GenerativeModel.from_cached_content(cached_content=cached_content, generation_config=generation_config, safety_settings=safety_settings)
    
    return model