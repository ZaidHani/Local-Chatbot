from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import os
from dotenv import load_dotenv
from threading import Thread
import time

load_dotenv()

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        wrapper.total_time += duration
        print(f"Execution time: {duration}")
        return result

    wrapper.total_time = 0
    return wrapper

local_chat = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    # THESE MODELS ARE VERY STUIPED, USE WITH CAUTION
    # model="deepseek-r1:1.5b",
    # model="gemma3:270m",
    model = 'deepseek-r1:1.5b-qwen-distill-q4_K_M',
    reasoning=False
)

groq_chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

def chat_with_groq(prompt: str):
    chat_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    formatted_prompt = chat_prompt.format_messages(input=prompt)
    response = groq_chat.invoke(formatted_prompt)
    return response.content

def chat_with_ollama(prompt: str):
    chat_prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    formatted_prompt = chat_prompt.format_messages(input=prompt)
    response = local_chat.invoke(formatted_prompt)
    return response.content