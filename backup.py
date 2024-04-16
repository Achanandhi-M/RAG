import os
import tempfile
import streamlit as st
from rag import ChatPDF
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
import urllib.request
import json

# Helper function to get available Ollama models from a server
@st.cache_data
def get_ollama_models(ollama_server_url):
    api_tags = "/api/tags"
    models = []
    with urllib.request.urlopen(ollama_server_url + api_tags) as tags:
        response = json.load(tags)
        models = [model['name'] for model in response['models']]
        models = [model.replace(":latest","") for model in models]
    return tuple(models)

# Callback handler for streaming Mistral chat responses to Streamlit
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.title("Private AI")
st.caption("Chat Locally with OpenSource LLMs like Mistral, llama2 etc..")

# Dropdown for selecting chat option
selected_option = st.sidebar.selectbox("Select Chat Option", ["Normal Chat", "Chat with PDF"])

if selected_option == "Normal Chat":
    with st.sidebar:
        # Url of the hosted ollama instance.
        st.sidebar.subheader("Chat With Open Source LLMs")
        ollama_server = st.text_input("Ollama API Server", value="http://localhost:11434")
        ollama_model = st.selectbox("Ollama model name", get_ollama_models(ollama_server), index=None)
        
        st.markdown("""
        Changing the model doesn't clear the history which gets passed to the new llm as context.  
        Use the below button to clear chat history and start with a clear slate.
        """)

        if st.button("Clear chat history", type="primary"):
            st.session_state["normal_chat_messages"] = [AIMessage(content="How can I help you?")]

    if "normal_chat_messages" not in st.session_state:
        st.session_state["normal_chat_messages"] = [AIMessage(content="How can I help you?")]

    for msg in st.session_state.normal_chat_messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input():
        st.session_state.normal_chat_messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)

        if not ollama_model:
            st.info("""Please select an existing Ollama model name to continue.
            Visit https://ollama.ai/library for a list of supported models.
            Restart the streamlit app after downloading a model using the `ollama pull <model_name>` command.
            It should become available in the list of available models.""")
            st.stop()

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            llm = ChatOllama(model=ollama_model, streaming=True, callbacks=[stream_handler])
            response = llm(st.session_state.normal_chat_messages)
            st.session_state.normal_chat_messages.append(AIMessage(content=response.content))

elif selected_option == "Chat with PDF":
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = None

    if "pdf_chat_messages" not in st.session_state:
        st.session_state["pdf_chat_messages"] = []  # Initialize session state key

    def read_and_save_file():
        st.session_state["pdf_chat_messages"] = []
        st.session_state["user_input"] = ""

        uploaded_files = st.session_state.get("file_uploader", [])

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.spinner(f"Ingesting {file.name}"):
                if st.session_state["assistant"] is not None:
                    st.session_state["assistant"].clear()
                    st.session_state["assistant"].ingest(file_path)
            os.remove(file_path)

    def display_messages():
        st.subheader("Chat")
        for i, msg in enumerate(st.session_state["pdf_chat_messages"]):
            if isinstance(msg, AIMessage):
                st.chat_message("assistant").write(msg.content)
            elif isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)

    def process_input():
        if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
            user_text = st.session_state["user_input"].strip()

            if st.session_state["assistant"] is not None:
                with st.spinner(f"Thinking"):
                    agent_text = st.session_state["assistant"].ask(user_text)

                st.session_state["pdf_chat_messages"].append((user_text, True))
                st.session_state["pdf_chat_messages"].append((agent_text, False))

    def page():
        if len(st.session_state) == 0:
            st.session_state["pdf_chat_messages"] = []
            st.session_state["assistant"] = ChatPDF()

        st.header("ChatPDF")

        st.subheader("Upload a document")
        st.file_uploader(
            "Upload document",
            type=["pdf"],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    if __name__ == "__main__":
        page()
