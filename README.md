# Legal Analysis Agent

This project showcases a powerful multi-agent system built with **LangGraph** for in-depth analysis of legal PDF documents. The agent categorizes user queries and routes them through a specialized workflow for research, analysis, and risk assessment, delivering a comprehensive final report.

The project is structured to be run in two primary ways: as an interactive web application using **Streamlit**, or as a robust API using **FastAPI**.

<img width="1278" height="1280" alt="Ekran görüntüsü 2025-07-15 152257" src="https://github.com/user-attachments/assets/b5125eca-7afd-4a55-a49d-a4b0220a4ed6" />


## ✨ Core Features

-   **Dynamic PDF Upload:** Analyze any PDF document on the fly.
-   **Multi-Agent Workflow:** Utilizes a graph-based system (LangGraph) where different "agents" handle specific tasks:
    -   **Categorizer:** Determines the user's intent.
    -   **Legal Researcher:** Gathers information from the document and the web.
    -   **Contract Analyst:** Extracts key clauses and obligations.
    -   **Risk Assessor:** Identifies potential risks and opportunities.
-   **Stateful Conversations (Streamlit):** Upload a document once and ask multiple follow-up questions in the same session.
-   **Powered by Groq & LLaMA 3.3:** Leverages the speed and power of the Groq LPU™ Inference Engine.
-   **Dual Interfaces:** Access the agent through a user-friendly web UI or a programmatic API.

## 🚀 How It Works

The core of the project is a `StateGraph` that manages the flow of information between different nodes. Each node is a specialized agent that performs a task and updates a shared state.

**High-Level Workflow:**
`User Query + PDF` → `[Categorizer]` → `[Agent Workflow (Research, Analysis, Risk)]` → `[Final Report]`

## 🛠️ Setup and Installation

Follow these steps to get the project running locally.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key**
    Create a file named `.env` in the root of the project directory and add your Groq API key:
    ```
    # .env
    GROQ_API_KEY="gsk_YourSecretGroqApiKeyHere"
    ```

## Usage

You can run the project in two ways:

### 1. Streamlit Web App (Recommended for demos)

This provides an interactive, chat-based interface.

1.  **Run the app:**
    ```bash
    streamlit run web_app.py
    ```
2.  Open your browser and navigate to the local URL provided (usually `http://localhost:8501`).
3.  Enter your API key, upload a PDF, and start asking questions!

<!-- 
Optional: You can uncomment and update this line to include a screenshot of your app.
![Streamlit App Screenshot](https://user-images.githubusercontent.com/12345/your-image-url.png) 
-->

### 2. FastAPI Server (For programmatic access)

This exposes the agent's functionality via an API endpoint.

1.  **Run the server:**
    ```bash
    uvicorn main:app --reload
    ```
2.  Open your browser and navigate to `http://127.0.0.1:8000/docs`.
3.  Use the interactive Swagger UI to upload a file and send queries to the `/analyze/` endpoint.
