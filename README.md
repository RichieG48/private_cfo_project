# Private CFO

A powerful AI assistant for financial analysis and decision-making.

## Prerequisites

- Python 3.10+
- Ollama installed and running

##  Getting Started

### 1. Model Setup (Local)

Pull the required models to your machine:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 2. Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/private-cfo.git
   ```

2. Navigate to the project directory:

   ```bash
   cd private-cfo
   ```

3. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

4. Activate the virtual environment (Windows):

   ```bash
   .venv\Scripts\activate
   ```

5. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

### 3. Environment Config

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

(Note: No API key is needed for the Local Agent)

### 4. Run the Application

Launch the "Cockpit" UI:

```bash
PYTHONPATH=. streamlit run src/app.py
```

## Usage Guide

1. Upload: Drag & Drop a PDF (e.g., Apple 10-K Report) into the sidebar.

2. Ingest: Click "Ingest & Memorize". The system will chunk, vectorize, and store the content locally.

3. Chat:

   - Ask a specific question: "What was the revenue in 2023?" -> Routed to Local Mistral.
   - Ask a general question: "Write a Python script to calculate CAGR." -> Routed to Cloud Gemini.

## Architectural Decisions

### Why Local Embeddings?

We use nomic-embed-text locally instead of OpenAI/Cohere embeddings. This ensures that even the mathematical representation of the private data remains on the user's hardware.

### The "Cookie Jar" Problem (Database Locking)

The system implements a custom Soft Reset mechanism for ChromaDB. Instead of deleting the database directory (which causes OS-level file locks in Streamlit), the application connects to the existing persistent client and wipes the collection via API. This ensures stable re-ingestion without crashing the UI.

I hope this helps! Let me know if you have any further questions.
