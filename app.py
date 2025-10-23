from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

chat = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="deepseek-r1:1.5b",
    temperature=0.0
)

print(chat.invoke('hello world').content)
