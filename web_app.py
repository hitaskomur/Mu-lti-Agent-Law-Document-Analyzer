# app.py

import os
import sys
import tempfile
import shutil
import pathlib
from typing import Dict, TypedDict  # DÜZELTİLDİ: TypedDict import edildi

# --- Streamlit Imports ---
import streamlit as st

# --- LangChain/LangGraph Imports ---
from langgraph.graph import StateGraph, END, START
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_groq import ChatGroq

# ==============================================================================
# 1. Page Configuration and Title
# ==============================================================================
st.set_page_config(page_title="Legal Analysis Agent", layout="wide")
st.title("📄 Legal Analysis Agent")
st.info(
    "This application uses a multi-agent workflow to analyze your uploaded Legal Law PDF document. "
    "Please enter your GROQ API key in the sidebar and upload a document to begin."
)

# ==============================================================================
# 2. The LawAgent Class (unchanged from previous logic)
# This class encapsulates the entire analysis workflow.
# ==============================================================================
class State(TypedDict):
    """Defines the state of the graph for a single analysis run."""
    query: str
    data: str
    category: str
    legal_research_response: str
    contract_analyst_response: str
    risk_assesment_response: str
    response: str

class LawAgent:
    """An agent that analyzes a single legal document by creating an isolated workflow."""
    def __init__(self, pdf_path: str, api_key: str):
        if not api_key:
            raise ValueError("GROQ API Key is required.")
        os.environ["GROQ_API_KEY"] = api_key
        
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.llm_with_tools = self.llm.bind_tools([DuckDuckGoSearchResults()])
        self.retriever, self.full_contract_text = self._setup_retriever(pdf_path)
        self.app = self._build_graph()

    def _setup_retriever(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
        return vectordb.as_retriever(), full_text

    def _create_qa_chain(self, llm_instance, prompt_template):
        return RetrievalQA.from_chain_type(llm=llm_instance, retriever=self.retriever, chain_type_kwargs={"prompt": prompt_template})

    # All node functions (categorize, legal_research, etc.) go here.
    # They are kept brief for readability but should be copied in full from the previous code.
    def categorize(self, state: State):
        prompt = ChatPromptTemplate.from_template("Categorize: {query}\nCategories: Contract Review, Legal Research, Risk Assesment, Custom Query.")
        chain = prompt | self.llm
        category_name = chain.invoke({"query": state["query"]}).content
        return {"category": category_name.strip().lower(), "data": self.full_contract_text}
    
    def legal_research(self, state: State) -> Dict[str, str]:
        prompt = PromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer:")
        qa_chain = self._create_qa_chain(self.llm_with_tools, prompt)
        result = qa_chain.invoke({"query": state["query"]})
        return {"legal_research_response": result["result"]}

    def contract_analyst(self, state: State) -> Dict[str, str]:
        prompt = PromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer:")
        qa_chain = self._create_qa_chain(self.llm, prompt)
        result = qa_chain.invoke({"query": state["query"]})
        return {"contract_analyst_response": result["result"]}

    def risk_assesment(self, state: State) -> Dict[str, str]:
        prompt = ChatPromptTemplate.from_template("Assess risks in this text: {data}")
        chain = prompt | self.llm
        result = chain.invoke({"data": state["data"]})
        return {"risk_assesment_response": result.content}

    def custom_query(self, state: State) -> Dict[str, str]:
        prompt = PromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer:")
        qa_chain = self._create_qa_chain(self.llm, prompt)
        result = qa_chain.invoke({"query": state["query"]})
        return {"response": result["result"]}

    def compliance_check(self, state: State) -> Dict[str, str]:
        prompt = ChatPromptTemplate.from_template("Combine insights into a final report:\n{legal_research_response}\n{contract_analyst_response}\n{risk_assesment_response}")
        chain = prompt | self.llm
        result = chain.invoke(state)
        return {"response": result.content}
    
    def _route_query(self, state: State):
        main_categories = ["contract review", "legal research", "risk assesment"]
        return "legal_research" if state["category"] in main_categories else "custom_query"

    def _build_graph(self):
        workflow = StateGraph(State)
        workflow.add_node("categorize", self.categorize)
        workflow.add_node("legal_research", self.legal_research)
        workflow.add_node("contract_analyst", self.contract_analyst)
        workflow.add_node("risk_assesment", self.risk_assesment)
        workflow.add_node("compliance_check", self.compliance_check)
        workflow.add_node("custom_query", self.custom_query)
        
        workflow.add_edge(START, "categorize")
        workflow.add_conditional_edges("categorize", self._route_query)
        workflow.add_edge("legal_research", "contract_analyst")
        workflow.add_edge("contract_analyst", "risk_assesment")
        workflow.add_edge("risk_assesment", "compliance_check")
        workflow.add_edge("compliance_check", END)
        workflow.add_edge("custom_query", END)
        return workflow.compile()

    def run(self, query: str):
        results = self.app.invoke({"query": query})
        return {
            "category": results.get("category", "unknown"),
            "response": results.get("response", "No response generated")
        }

# ==============================================================================
# 3. Sidebar for Configuration and File Upload
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key input
    api_key = st.text_input("GROQ API Key", type="password", placeholder="gsk_...")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

    # Session reset button
    if st.button("Upload New Document / Reset"):
        # Clear the session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ==============================================================================
# 4. Session State Management
# ==============================================================================
# Initialize required keys in session state if they don't exist
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# Initialize the agent only when a new file is uploaded and an API key is provided
if uploaded_file and api_key:
    # If there's no agent or the uploaded file has changed, create a new agent
    if st.session_state.agent is None or st.session_state.uploaded_file_name != uploaded_file.name:
        with st.spinner(f"Preparing '{uploaded_file.name}' for analysis... This may take a moment."):
            try:
                # Save the file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    shutil.copyfileobj(uploaded_file, tmp_file)
                    tmp_file_path = tmp_file.name
                
                # Initialize the agent
                st.session_state.agent = LawAgent(pdf_path=tmp_file_path, api_key=api_key)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.chat_history = []  # Reset chat history for the new file
                os.remove(tmp_file_path) # Clean up the temporary file
                st.sidebar.success(f"'{uploaded_file.name}' is ready for analysis!")
            except Exception as e:
                st.sidebar.error(f"Error initializing agent: {e}")
                st.session_state.agent = None # Reset agent on error
else:
    # Reset the agent if the file or API key is removed
    if not uploaded_file or not api_key:
        st.session_state.agent = None
        st.session_state.uploaded_file_name = None

# ==============================================================================
# 5. Chat Interface
# ==============================================================================
# Display the chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new input from the user
if prompt := st.chat_input("Ask a question about the document..."):
    # First, check if the prerequisites are met
    if not api_key:
        st.warning("Please enter your GROQ API key to begin.")
    elif st.session_state.agent is None:
        st.warning("Please upload a PDF document to analyze.")
    else:
        # Add user's message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.run(query=prompt)
                    response = result["response"]
                    st.markdown(response)
                    # Add assistant's response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})