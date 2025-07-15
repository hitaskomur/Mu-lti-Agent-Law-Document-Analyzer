# main.py

import os
import sys
import shutil
import tempfile
from typing import Dict, TypedDict

# --- FastAPI Imports ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- LangChain/LangGraph Imports (Önceki koddan) ---
from langgraph.graph import StateGraph, END, START
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_groq import ChatGroq
import pathlib

# --- Environment Variable Loading (Optional but Recommended) ---
from dotenv import load_dotenv
load_dotenv() # .env dosyasından ortam değişkenlerini yükler



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please create a .env file or set it manually.")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Diğer yapılandırmalar
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

app = FastAPI(
    title="Legal Analysis Agent API",
    description="Upload a PDF document and ask questions about it. The API will perform a multi-step analysis.",
    version="1.0.0",
)

# ==============================================================================
# 2. Pydantic Models for API
# API'nin girdi ve çıktılarını tanımlar (dökümantasyon için harika!)
# ==============================================================================
class AnalysisResponse(BaseModel):
    category: str
    response: str
    
# ==============================================================================
# 3. LawAgent Sınıfı (Önceki koddan aynen alındı)
# Bu sınıf, tüm analiz mantığını kendi içinde barındırır.
# ==============================================================================
class State(TypedDict):
    """Grafiğin durumunu tanımlar."""
    query: str
    data: str
    category: str
    legal_research_response: str
    contract_analyst_response: str
    risk_assesment_response: str
    response: str

class LawAgent:
    """Tek bir yasal belgeyi analiz eden ve izole bir iş akışı yürüten ajan."""
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The specified file does not exist: {pdf_path}")
        self.llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0)
        self.llm_with_tools = self.llm.bind_tools([DuckDuckGoSearchResults()])
        self.retriever, self.full_contract_text = self._setup_retriever(pdf_path)
        self.app = self._build_graph()

    def _setup_retriever(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        full_text = "\n".join([doc.page_content for doc in docs])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
        return vectordb.as_retriever(), full_text

    def _create_qa_chain(self, llm_instance, prompt_template):
        return RetrievalQA.from_chain_type(llm=llm_instance, retriever=self.retriever, chain_type_kwargs={"prompt": prompt_template})

    def categorize(self, state: State):
        prompt = ChatPromptTemplate.from_template("Categorize the query: {query}\nCategories: Contract Review, Legal Research, Risk Assesment, Custom Query. Only name.")
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
    

    def run(self, query: str):
        inputs = {"query": query}
        results = self.app.invoke(inputs)
        return {
            "category": results.get("category", "unknown"),
            "response": results.get("response", "No response generated")
        }
# ==============================================================================
# 4. API ENDPOINT
# ==============================================================================

@app.post("/analyze/", response_model=AnalysisResponse)
def analyze_document(
    query: str = Form(..., description="The question you want to ask about the document."),
    file: UploadFile = File(..., description="The PDF document to analyze.")
):
    """
    Analyzes an uploaded PDF document based on a user query.
    
    This endpoint uses a temporary directory to securely handle file uploads,
    preventing file locking issues common on Windows.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    # with bloğu sayesinde, bu bloktan çıkıldığında (hata olsa bile) 
    # temp_dir ve içindeki her şey otomatik olarak ve güvenli bir şekilde silinir.
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Geçici dizin içinde dosya için bir yol oluştur.
            temp_pdf_path = pathlib.Path(temp_dir) / file.filename
            
            # Yüklenen dosyayı bu yeni yola kaydet.
            with open(temp_pdf_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)
            
            # Şimdi LawAgent'ı bu güvenli, geçici dosya yoluyla başlat.
            print(f"Processing temporary file: {temp_pdf_path}")
            agent = LawAgent(pdf_path=str(temp_pdf_path))
            
            # Analizi çalıştır.
            result = agent.run(query=query)
            
            return JSONResponse(content=result)

    except FileNotFoundError as e:
        # LawAgent başlatılırken dosya bulunamazsa
        print(f"File not found during agent initialization: {e}", file=sys.stderr)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Diğer beklenmedik hataları yakala.
        print(f"An internal error occurred: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"An internal error occurred while processing the document: {str(e)}")

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the Legal Analysis Agent API. Go to /docs to see the API documentation."}


# ==============================================================================
# 5. Uvicorn Server
# Bu blok, dosyanın `python main.py` ile çalıştırılmasını sağlar.
# ==============================================================================
if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)