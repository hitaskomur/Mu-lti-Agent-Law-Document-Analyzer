# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import sys
from typing import Dict, TypedDict

# Third-party imports for core functionality
from langgraph.graph import StateGraph, END, START

# LangChain specific imports
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_groq import ChatGroq

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
# WARNING: Do not hardcode API keys in production. Use environment variables.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ==============================================================================
# 3. GRAPH STATE DEFINITION
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

# ==============================================================================
# 4. THE LAW AGENT CLASS
# This class encapsulates the entire workflow for a single PDF document.
# ==============================================================================
class LawAgent:
    """
    An agent that analyzes a single legal document by creating an isolated,
    in-memory vector store and processing queries through a stateful graph.
    """
    def __init__(self, pdf_path: str):
        """
        Initializes the agent for a specific PDF file.

        Args:
            pdf_path (str): The path to the PDF file to be analyzed.
        """
        print(f"Initializing agent for: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The specified file does not exist: {pdf_path}")
            
        # Initialize LLMs and tools
        self.llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0)
        self.tools = [DuckDuckGoSearchResults()]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Setup the retriever and get full text, specific to the provided PDF
        self.retriever, self.full_contract_text = self._setup_retriever(pdf_path)
        
        # Build the graph and compile the runnable app
        self.app = self._build_graph()
        print("Agent initialized successfully.")

    def _setup_retriever(self, pdf_path: str):
        """
        Creates an isolated, in-memory vector store for the given PDF.
        This ensures no data from previous runs is used.
        """
        print("Loading document and creating in-memory vector store...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        full_text = "\n".join([doc.page_content for doc in docs])
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Create an in-memory vector store by NOT providing a persist_directory.
        # This ensures data isolation for each run.
        vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
        
        print("In-memory vector store created.")
        return vectordb.as_retriever(), full_text

    def _create_qa_chain(self, llm_instance, prompt_template):
        """Helper method to create a RetrievalQA chain."""
        return RetrievalQA.from_chain_type(
            llm=llm_instance,
            retriever=self.retriever, # Uses the instance-specific retriever
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )

    # --- Graph Node Methods ---
    
    def categorize(self, state: State) -> Dict[str, str]:
        """Node: Categorize the user query and inject the full text into the state."""
        print("--- Node: categorize ---")
        prompt = ChatPromptTemplate.from_template(
            "Categorize the query into: Contract Review, Legal Research, Risk Assesment, or Custom Query.\n"
            "Query: {query}\n\nJust return the category name."
        )
        chain = prompt | self.llm
        category_name = chain.invoke({"query": state["query"]}).content
        return {"category": category_name.strip().lower(), "data": self.full_contract_text}

    def legal_research(self, state: State) -> Dict[str, str]:
        """Node: Perform legal research using the vector store and web search."""
        print("--- Node: legal_research ---")
        prompt = PromptTemplate.from_template(
            "Use context and web search to find legal cases, regulations, and citations. "
            "If you don't know the answer, say so.\n"
            "Context: {context}\nQuestion: {question}\nHelpful Answer:"
        )
        qa_chain = self._create_qa_chain(self.llm_with_tools, prompt)
        result = qa_chain.invoke({"query": state["query"]})
        return {"legal_research_response": result["result"]}

    def contract_analyst(self, state: State) -> Dict[str, str]:
        """Node: Analyze the contract for key clauses, risks, and obligations."""
        print("--- Node: contract_analyst ---")
        prompt = PromptTemplate.from_template(
            "Analyze the contract for key clauses, risks, and obligations using the context. "
            "If you don't know, say so.\n"
            "Context: {context}\nQuestion: {question}\nHelpful Answer:"
        )
        qa_chain = self._create_qa_chain(self.llm, prompt)
        result = qa_chain.invoke({"query": state["query"]})
        return {"contract_analyst_response": result["result"]}

    def risk_assesment(self, state: State) -> Dict[str, str]:
        """Node: Assess the contract for legal risks using the full text."""
        print("--- Node: risk_assesment ---")
        prompt = ChatPromptTemplate.from_template(
            "Using the full contract text, assess for legal risks and opportunities. "
            "Provide actionable recommendations.\n"
            "Full Contract Text:\n{data}\nHelpful Answer:"
        )
        chain = prompt | self.llm
        result = chain.invoke({"data": state["data"]})
        return {"risk_assesment_response": result.content}

    def custom_query(self, state: State) -> Dict[str, str]:
        """Node: Handle any query that doesn't fit the main categories."""
        print("--- Node: custom_query ---")
        prompt = PromptTemplate.from_template(
            "Answer the question concisely using the context. "
            "If you don't know, say so. Always say 'thanks for asking!' at the end.\n"
            "Context: {context}\nQuestion: {question}\nHelpful Answer:"
        )
        qa_chain = self._create_qa_chain(self.llm, prompt)
        result = qa_chain.invoke({"query": state["query"]})
        return {"response": result["result"]}

    def compliance_check(self, state: State) -> Dict[str, str]:
        """Node: Combine all previous analysis into a final, comprehensive report."""
        print("--- Node: compliance_check ---")
        prompt = ChatPromptTemplate.from_template(
            """Combine and summarize all insights provided by the Legal Researcher, Contract Analyst, and Legal Strategist into a comprehensive report.
            Ensure the final report includes references to all relevant sections from the document.
            Provide actionable recommendations and ensure compliance with application laws.
            Summarize and integrate the following insight gathered using the full contract data:
            "--- INSIGHTS ---\n"
            "Legal Research:\n{legal_research_response}\n\n"
            "Contract Analysis:\n{contract_analyst_response}\n\n"
            "Risk Assessment:\n{risk_assesment_response}\n\n"
            Provide a structured legal analysis report that includes key terms, obligations, risks, and recommendation, with references to the document.
            "--- FINAL REPORT ---"""
        )
        chain = prompt | self.llm
        result = chain.invoke(state) # Pass the whole state dict
        return {"response": result.content}
        
    def _route_query(self, state: State) -> str:
        """Conditional Edge: Route the query based on its category."""
        print(f"Routing based on category: '{state['category']}'")
        main_categories = ["contract review", "legal research", "risk assesment"]
        return "legal_research" if state["category"] in main_categories else "custom_query"

    def _build_graph(self):
        """Builds and compiles the LangGraph workflow."""
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

    def run(self, query: str) -> Dict[str, str]:
        """
        Processes a user query through the compiled LangGraph workflow.

        Args:
            query (str): The user's query.

        Returns:
            A dictionary containing the final results.
        """
        inputs = {"query": query}
        print("\nInvoking the agent workflow...")
        results = self.app.invoke(inputs)
        return {
            "category": results.get("category", "unknown"),
            "response": results.get("response", "No response generated")
        }


# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # --- Step 1: Get PDF path from user ---
    pdf_file_path = input("Please enter the full path to your PDF file: ").strip()
    
    # Simple validation
    if not pdf_file_path.lower().endswith('.pdf'):
        print("Error: Please provide a valid .pdf file.")
        sys.exit(1)

    try:
        # --- Step 2: Initialize the agent for the specified file ---
        agent = LawAgent(pdf_path=pdf_file_path)
        
        # --- Step 3: Get query from user ---
        user_query = input("What would you like to know about this document? ")
        
        # --- Step 4: Run the analysis ---
        final_result = agent.run(query=user_query)

        # --- Step 5: Print the final report ---
        print("\n" + "="*50)
        print("           AGENT EXECUTION COMPLETE")
        print("="*50 + "\n")
        print(f"Initial Query: {user_query}\n")
        print(f"Detected Category: {final_result['category']}\n")
        print("--- FINAL REPORT ---")
        print(final_result['response'])
        print("\n" + "="*50)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)