from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. always, ALWAYS provide the context in your answer

context: {context}

question: {query}

answer:
"""

custom_rag_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
