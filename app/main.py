import streamlit as st
import requests
import json
from langdetect import detect

API_URL = "http://localhost:8000/ask"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = "test-session" 

st.title("Bangla Textbook RAG Chatbot")
st.caption("Ask questions in either English or Bangla")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question (English or Bangla)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        response = requests.post(
            API_URL,
            params={"session_id": st.session_state.session_id}, 
            json={"query": prompt},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        api_response = response.json()
        answer = api_response.get("answer", "No response found")
        
    except Exception as e:
        answer = f"Error: {str(e)}"
    
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.messages:
    try:
        last_message = st.session_state.messages[-1]["content"]
        lang = detect(last_message)
        st.sidebar.markdown(f"**Last response language:** {'Bangla' if lang == 'bn' else 'English'}")
    except:
        pass