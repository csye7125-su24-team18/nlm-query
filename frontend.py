import streamlit as st
import requests
import os

FASTAPI_HOST = os.getenv("HOST", "localhost")
FASTAPI_PORT = int(os.getenv("PORT", 8000))
API_ENDPOINT = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/ask"

st.title("Chat with CVE Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

def chat():
    user_input = st.text_input("Enter your query:", key="input", placeholder="Type your message here...", label_visibility="collapsed")

    if st.button("Send"):
        if user_input:
            st.session_state.messages.append({"user": user_input})
            
            try:
                response = requests.post(API_ENDPOINT, json={"question": user_input})
                if response.status_code == 200:
                    bot_response = response.json().get("answer", "Sorry, I couldn't find an answer.")
                else:
                    bot_response = "Error: Could not fetch response from the server."
            except requests.exceptions.RequestException as e:
                bot_response = f"Error: {str(e)}"

            st.session_state.messages.append({"bot": bot_response})
            st.experimental_rerun()

# Display chat messages
for message in st.session_state.messages:
    if "user" in message:
        st.markdown(f"**You:** {message['user']}")
    elif "bot" in message:
        st.markdown(f"**Bot:** {message['bot']}")

chat()
