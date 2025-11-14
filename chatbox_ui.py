import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_db"
vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding_function(),)
height = 600
title = "Chatbox UI"
icon = ":university:"

st.set_page_config(page_title=title, page_icon=icon)

st.header(title)

message = st.container(border = True,height = height)

if "messages" not in st.session_state:
    st.session_state.messages = []

    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks. "))

 # display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("How are you?")

# did the user submit a prompt?
if prompt:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)

        st.session_state.messages.append(HumanMessage(prompt))

    model = Ollama(model="gemma:7b")
    # creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.75},
    )

    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # creating the system prompt
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Context: {context}:"""

    # Populate the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)


    print("-- SYS PROMPT --")
    print(system_prompt_fmt)

    # adding the system prompt to the message history
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # invoking the llm
    result = model.invoke(st.session_state.messages)

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append(AIMessage(result))