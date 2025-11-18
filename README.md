# UniScout RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) assistant designed to help users with university research. It leverages local Large Language Models (LLMs) and embeddings, powered by **Ollama**, to provide answers and recommendations based on university brochures stored locally.

## 1. Prerequisites

You must have the following core software installed and configured before running the project.

### 1.1. Core Software

| Requirement | Description | Installation Link |
| :--- | :--- | :--- |
| **Python** | Version 3.10 or higher. | [Python Downloads](https://www.python.org/downloads/) |
| **Ollama** | The local server application for running LLMs and embedding models. **This must be running** for the RAG system to work. | [Ollama Installation](https://ollama.com/download) |

## 2. Setup and Ollama Configuration

### 2.1. Install Python Libraries

Use a virtual environment to manage project dependencies.

```bash
# 1. Create a virtual environment (if one doesn't exist)
python3 -m venv venv

# 2. Activate the environment
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate # Windows (PowerShell)

# 3. Install all required packages
pip install streamlit==1.32.0 \
            pypdf \
            langchain-core \
            langchain-community \
            chromadb

```

## 2.2. Configure and Start Ollama

The system requires the Ollama server to be running and two specific models to be downloaded.

A. Start the Ollama Server

The Ollama server must be actively running and accessible on http://localhost:11434.
    If using the macOS/Windows App: Ensure the Ollama desktop application (the llama icon in your menu bar/system tray) is running. The server starts automatically.
    If using the command line (Linux/Manual): Open a dedicated terminal window and run the following command. Keep this terminal open for the entire duration of your session.

```bash
ollama serve
```
B. Pull the Required Models

The RAG application relies on these two models. Run these commands in a separate terminal once Ollama is running.
```bash
ollama pull mxbai-embed-large
ollama pull gemma:7b
ollama pull llama3.1:latest
```

 ## 3. Running the UniScout Assistant

The project is executed in two simple steps.

### Step 3.1: Create the Vector Database

The populate_database.py script loads your PDF files, splits them into chunks, embeds them using mxbai-embed-large, and saves the searchable index. This step only needs to be run once, or whenever you add new brochures to the ./data folder. 
Ensure all PDFs are placed inside the ./data directory.
Run the population script:    
```bash
python populate_database.py
```

### Step 3.2: Launch the Streamlit Interface

Once the database is ready, you can start the interactive web interface.
Ensure the Ollama server (ollama serve or the desktop app) is still running.
Run the Streamlit application:

```bash
streamlit run interface3.py
```    

## 4. Description of the files 

| File                                   | Description                                                                                                                                                                                                                                         |
| :------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **populate_database.py**               | Loads all PDF brochures from the `./data` folder, splits them into text chunks, generates embeddings using **mxbai-embed-large**, and stores them in a Chroma vector database for retrieval. |
| **get_embedding_function.py**          | Defines and initializes the embedding model used for converting text into vector representations. It provides the embedding function to other modules (e.g., for populating or querying the database).                                              |
| **query_data.py**                      | First mode of the solution, where students can ask specific information about a university. Handles queries to the vector database. It retrieves the most relevant text chunks based on a user’s question using semantic similarity search and prepares the context for the language model. When calling it the university must be specified in the query.                                                    |
| **comparing.py**                       | Provides functions to compare universities on specific information. Need to specify which uni to compare in the query.                                                                               |
| **final_recomm.py**                    | Implements the logic for generating tailored university recommendations. It combines query results and LLM reasoning (via Ollama) to produce the final ranked suggestions.                                                                          |
| **interface3.py**                      | The main Streamlit web application file. It runs the interactive interface where users can type queries, receive results, and view recommendations in real time.                                                                                    |
| **sample_question_recommendation.txt** | Contains example user queries for the different modes. Used for testing and demonstration.                                                                |

## 5. File structure 

AI_Project/
│
├── data/                      # PDF brochures
├── chroma/                    # Vector database storage
├── interface3.py              # Streamlit front-end
├── populate_database.py       # Embedding + database population
├── query_data.py              # Data retrieval logic
├── final_recomm.py            # Recommendation generation
├── get_embedding_function.py  # Embedding setup
├── comparing.py               # Similarity and comparison
└── sample_question_recommendation.txt  # Example inputs

## 6. Credits and Acknowledgments 
Developed by Edoardo Ciato, Alejandra Rabattu, Juan Perez, Diego Martinez, and Sibylle Radix for AI-MACHINE LEARNING FOUNDATIONS, using Ollama, LangChain, and Streamlit.
