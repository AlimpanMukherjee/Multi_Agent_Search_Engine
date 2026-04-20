import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq

st.set_page_config(page_title="Search Agent")
st.title("🔍 Search Agent")
st.caption("Search Wikipedia, Arxiv and LangSmith docs")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API key:", type="password")
    st.markdown("---")


if not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar to get started.")
    st.stop()

# Initialize everything once
@st.cache_resource
def initialize_agent(api_key):
    # LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

    # Tools
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

    loader = WebBaseLoader("https://docs.smith.langchain.com/")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    vectordb = Chroma.from_documents(documents, HuggingFaceEmbeddings())
    retriever = vectordb.as_retriever()
    retriever_tool = create_retriever_tool(retriever, "langsmith-search", "Search LangSmith documentation")

    tools = [wiki, arxiv, retriever_tool]

    # Prompt
    prompt = PromptTemplate.from_template("""You are a helpful assistant. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Chat History: {chat_history}
Question: {input}
Thought: {agent_scratchpad}""")

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor

# Session state
if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Load agent
with st.spinner("Loading agent and tools..."):
    agent_executor = initialize_agent(groq_api_key)

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    from langchain.callbacks import StreamlitCallbackHandler
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent_with_history.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": "default_session"},
                "callbacks": [st_cb]
            }
        )
        answer = response["output"]
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})