import os
from dotenv import load_dotenv
import google.generativeai as genai


def configure_GEMINI_API():
    
    load_dotenv()
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)