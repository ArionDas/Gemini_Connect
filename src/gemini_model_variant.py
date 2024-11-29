import os
from dotenv import load_dotenv
import google.generativeai as genai


def choose_GEMINI_model_variant(model_number : int) : 
    
    gemini_model_variants = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    
    gemini_variant = gemini_model_variants[model_number]
    model = genai.GenerativeModel(gemini_variant)
    
    return model