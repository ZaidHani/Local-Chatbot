from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from base64 import b64decode
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.page_content

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), model='llama-3.3-70b-versatile') 
# llm = ChatOllama(model="llava:latest", base_url=os.getenv("OLLAMA_BASE_URL"))

embeddings = OllamaEmbeddings(
    model="embeddinggemma:latest", 
    base_url=os.getenv("OLLAMA_BASE_URL")
    )

vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings, persist_directory='./data/chroma_db')
retriever = vectorstore.as_retriever()

chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | llm
    | StrOutputParser()
)

chain_with_sources = {
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | llm
        | StrOutputParser()
    )
)