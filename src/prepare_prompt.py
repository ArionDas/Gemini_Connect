import os
from dotenv import load_dotenv
import google.generativeai as genai

from dataset import get_title_names, get_summaries


def prepare_USER_prompt(user_query) : 
    
    summaries = get_summaries()
    
    prompt = f"User's query : {user_query}\n \
                Please return the titles of the most relevant papers to the user query.\
                Here are the summaries of some paper for you to refer to : \n \
              {summaries[:1450]}."

    return prompt