# eBay Policy Assistant – RAG Chatbot with Streaming Responses

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on a provided eBay policy document (T&Cs and User Agreement). It uses semantic search, a vector database, and an instruction-tuned LLM to provide accurate, real-time responses via a Streamlit interface.

---

## Project Architecture & Flow


                         ┌────────────┐
                         │  Raw PDF   │
                         └─────┬──────┘
                               ↓
                 [Text Cleaning & Header/Footer Removal]
                               ↓
                    ┌────────────────────────┐
                    │  cleaned_text.txt      │
                    └────────┬───────────────┘
                             ↓
               [Sentence-Aware Chunking (spaCy)]
                             ↓
                    ┌────────────────────────┐
                    │ document_chunks.txt    │
                    └────────┬───────────────┘
                             ↓
             [Embeddings (bge-small-en) + FAISS Index]
                             ↓
         ┌──────────────────────────────┐
         │ faiss_index.idx + chunks.pkl │
         └────────────┬─────────────────┘
                      ↓
        [Query ➝ Top-K Retrieval ➝ Prompt Injection]
                      ↓
          [LLM Response via HuggingFace Inference API]
                      ↓
               [Real-Time Streaming in Streamlit UI]


---

## Project Structure

.
├── app.py                       # Streamlit chatbot app with streaming response
├── README.md                    # Project overview and instructions
├── requirements.txt             # Required Python packages
├── /data                        # Raw and cleaned input documents
│   ├── AI_Training_Document.pdf
│   └── cleaned_text.txt
├── /chunks                     # Chunked text segments
│   └── document_chunks.txt
├── /vectordb                   # FAISS index and chunk metadata
│   ├── faiss_index.idx
│   └── chunks.pkl
├── /notebooks                  # Preprocessing scripts
│   ├── extract_text_from_pdf.py      # Extract and clean raw PDF
│   ├── chunk_text.py                 # Sentence-aware chunking using spaCy
│   └── create_embeddings.py          # Generate embeddings and build FAISS index
├── /src                        # RAG pipeline logic
│   ├── rag_pipeline.py               # Class to retrieve and generate responses
│   └── test_rag_pipeline.py          # CLI-based test runner for the pipeline
└── .env (not uploaded)         # Stores HuggingFace API token (used internally)

---

## Components Explained

---

1. **Document Preprocessing**
- **File:** `extract_text_from_pdf.py`  
- Cleans headers/footers like "eBay" or "Page X"  
- Saves cleaned output as `cleaned_text.txt`

---

2. **Sentence-Aware Chunking**
- **File:** `chunk_text.py`  
- Uses **spaCy** to split sentences into ~150-word chunks  
- Output is saved in `chunks/document_chunks.txt`

---

3. **Embeddings & Vector Index**
- **File:** `create_embeddings.py`  
- Uses **BAAI/bge-small-en** to generate sentence embeddings  
- Embeddings are indexed using **FAISS**  
- Output: `faiss_index.idx` and `chunks.pkl` saved in `/vectordb/`

---

4. **RAG Pipeline (Retriever + Generator)**
- **File:** `src/rag_pipeline.py`  
- Loads the **FAISS** vector database and retrieves relevant chunks  
- Injects them into a prompt template for the LLM  
- Calls the **Meta-LLaMA-3-8B-Instruct** model from **Hugging Face**  
- Streams responses in real-time  
- Uses a `.env` file (not shared) to store the **HF_TOKEN** securely

---

5. **Streaming Chatbot Interface**
- **File:** `app.py`  
- Built using **Streamlit**  
- **Key Features:**  
  - Real-time token-level streaming  
  - Displays model info and source highlights  
  - Option to clear/reset chat

---

## How to Run the Project Locally

**Clone the Repository**

git clone https://github.com/your-username/rag-chatbot-ebay-policy.git
cd rag-chatbot-ebay-policy

**Create & Activate a Virtual Environment** (recommended to avoid conflicts)

- python -m venv venv
- On Windows: .venv\Scripts\activate
- macOS/Linux: source .venv/bin/activate

**Install Dependencies**

pip install -r requirements.txt

**Run Preprocessing Steps**

cd notebooks
python extract_text_from_pdf.py
python chunk_text.py
python create_embeddings.py
cd ..

**Launch the Streamlit App**

streamlit run app.py

---

## Model & Embedding Choices

| Component        | Model Used               |
|------------------|--------------------------|
| Embedding Model | BAAI/bge-small-en        |
| LLM for Q&A      | Meta-LLaMA-3-8B-Instruct |
| Vector DB        | FAISS                    |

---

## 🎥 Demo

| Type     | Link                                                                 |
|----------|----------------------------------------------------------------------|
| 🔗 GitHub | [rag-chatbot-ebay-policy](https://github.com/TinaSingh19/rag-chatbot-ebay-policy) |
| 📹 Video  | [Google Drive Demo Folder](https://drive.google.com/drive/folders/1MQlx_g6MX64-QhzMmVtoBxD2h3YzRE3P?usp=sharing) |






