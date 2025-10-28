import streamlit as st
import chat
import prompt
import vectorizor
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

vector_store = FAISS.load_local("./data/embeddings/", embeddings=vectorizor.embedding_model, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever()

llm = chat.groq_chat

rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt.custom_rag_prompt
    | llm
    | StrOutputParser()
)

st.write(rag_chain.invoke("التغطيات داخل المستشفى"))
