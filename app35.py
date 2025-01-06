import streamlit as st
from langchain.schema import HumanMessage, AIMessage  # Importing HumanMessage and AIMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import io
import numpy as np

# Streamlit app configuration
st.set_page_config(page_title="Prompt Engineering Assistant", page_icon="🤖")

# Title and greeting message
st.title('AI Prompt Engineering Assistant')
st.markdown("Hello! I'm your AI assistant. I can help you with prompt engineering training and assist in complex queries.")

# User level selection (Beginner, Intermediate, Expert)
user_level = st.selectbox("Select your proficiency level:", ["Beginner", "Intermediate", "Expert"])

# Define modular prompts based on user level
level_prompts = {
    "Beginner": "You are a beginner in prompt engineering. Please explain the basics and help with simple queries.",
    "Intermediate": "You have some knowledge of prompt engineering. Guide me through more complex prompts and strategies.",
    "Expert": "You are an expert in prompt engineering. Assist me with advanced techniques and strategies for multi-step tasks."
}

# AI assistant prompt template for training
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(level_prompts[user_level]),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# Chat message history initialization
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Setup AI model
api_key = "AIzaSyCX_kUiDrld68GV3SI4poivKsP6Xz1rkIk"  # Enter your Google API key here
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Create the processing chain for prompt engineering
chain = prompt | model

# Combine chain with message history for continuous learning
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# User input for prompt engineering or complex queries
user_input = st.text_input("Enter your prompt or query:", "")

if user_input:
    st.chat_message("human").write(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        config = {"configurable": {"session_id": "any"}}

        # Use user input to get AI response (training or query assistance)
        response = chain_with_history.stream({"question": user_input}, config)

        # Process response in chunks
        for res in response:
            if isinstance(res, str):
                content = res
            elif hasattr(res, 'content'):
                content = res.content
            else:
                st.error("Unexpected response format.")
                continue
            
            # Update response incrementally
            full_response += content
            message_placeholder.markdown(full_response)

else:
    st.warning("Please enter your question.")

# Multimodal support for uploading images/videos
uploaded_file = st.file_uploader("Upload an image or video (optional):", type=["png", "jpg", "jpeg", "mp4", "avi", "mpeg4"])

if uploaded_file is not None:
    # Check if the file is an image or video
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension in ['png', 'jpg', 'jpeg']:
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated to use `use_container_width`
        st.write("Processing image...")

        # Example: Simple image processing (you can replace this with an AI model like Google Vision)
        st.write("Image processed successfully.")

    elif file_extension in ['mp4', 'avi', 'mpeg4']:
        # Process video
        st.video(uploaded_file)
        st.write("Processing video...")

        # Example: Simple video processing (you can replace this with an AI model for video analysis)
        st.write("Video processed successfully.")

    else:
        st.error("Unsupported file type. Please upload a PNG, JPG, JPEG image or MP4, AVI, MPEG4 video.")

# User progress tracking (session history)
if st.button("Show chat history"):
    chat_history = msgs.messages  # Accessing the messages directly
    if chat_history:
        for message in chat_history:
            # Display messages bpased on their type
            if isinstance(message, HumanMessage):
                st.write(f"Human: {message.content}")  # Explicitly label human messages
            elif isinstance(message, AIMessage):
                st.write(f"AI: {message.content}")  # Explicitly label AI messages
            else:
                st.write("Unknown message type.")  # Handle unknown message types
    else:
        st.write("No chat history available.")
    