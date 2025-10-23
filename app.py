from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

chat = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="gemma3:270m",
    temperature=0.0
)

print(chat.invoke('hello world').content)