import streamlit as st
from openai import OpenAI
import os

# Page configuration
st.set_page_config(
    page_title="AI Supervisor",
    page_icon="💬",
    layout="centered"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main-header {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #1E88E5 !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #E3F2FD;
        border-bottom-right-radius: 0.2rem;
    }
    .chat-message.assistant {
        background-color: #F5F5F5;
        border-bottom-left-radius: 0.2rem;
    }
    .chat-message .message-content {
        margin-left: 10px;
        margin-right: 10px;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        padding: 1rem;
        background-color: white;
        width: 100%;
        left: 0;
    }
    .stTextInput > div {
        padding-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        border-radius: 20px !important;
        padding: 10px 15px !important;
        border: 1px solid #E0E0E0 !important;
    }
    .footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">💬 Your AI Supervisor</h1>', unsafe_allow_html=True)

# Fetch API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# If key is not found, show an error
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables.", icon="🚫")
else:
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Set up session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask me anything...")

    # Handle user input
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show user message immediately
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            # Stream response from OpenAI
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )

            # Process the streaming response
            for chunk in stream:
                if chunk_content := chunk.choices[0].delta.content:
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a simple clear button
if st.session_state.get("messages") and len(st.session_state.messages) > 0:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()