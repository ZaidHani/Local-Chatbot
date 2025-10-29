import streamlit as st
import time
from chat import chain

st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🤖")
st.title("Multimodal RAG Chatbot")

if "chat_history" not in st.session_state:
	st.session_state.chat_history = []

st.write("Ask a question about your documents. Responses use multimodal RAG (text, tables, images).")

user_input = st.text_input("You:", key="user_input")
submit = st.button("Send")

if submit and user_input:
	with st.spinner("Thinking..."):
		begin_time = time.time()
		response = chain.invoke(user_input)
		st.write("Bot:", response)
		end_time = time.time()
	st.write(f"_Response time: {end_time - begin_time:.2f} seconds_")