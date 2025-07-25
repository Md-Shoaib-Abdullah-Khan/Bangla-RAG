import streamlit as st
import requests


API_URL = "http://localhost:8000/ask" 


if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("Bangla RAG Chatbot")
st.caption("Ask about Kalyanee's age during marriage")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    try:
        response = requests.post(
            API_URL,
            json={"query": prompt},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        answer = response.json()
        
    except Exception as e:
        answer = f"Error: {str(e)}"
    
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})