import streamlit as st
import chat
import vectorizor
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

vector_store = FAISS.load_local("./data/embeddings/", embeddings=vectorizor.embedding_model, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever()

llm = chat.local_chat

chain = llm | retriever

st.title("Local Chatbot with Streamlit")
user_input = st.text_input("You: ")
if user_input:
    response = chain.invoke({"input": user_input})
    st.write("Bot:", response)