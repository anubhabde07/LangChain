import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.title("🤖 LangChain Chatbot")

# Load model only once
if "chat_model" not in st.session_state:
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=200,
        temperature=0.7
    )
    st.session_state.chat_model = ChatHuggingFace(llm=llm)

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Convert to LangChain message format
    lc_history = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))

    response = st.session_state.chat_model.invoke(lc_history)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )

    st.chat_message("assistant").write(response.content)
