import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class RAGPipeline:
    def __init__(self, vectordb_path: str, embedding_model: str = "BAAI/bge-small-en"):
        """
        Initializes the Retrieval-Augmented Generation (RAG) pipeline:
        - Loads FAISS index and metadata
        - Loads the sentence transformer embedding model
        - Initializes the Hugging Face Inference Client for Meta LLaMA-3-8B-Instruct
        """
        self.vectordb_path = vectordb_path
        self.embedding_model = SentenceTransformer(embedding_model)

        print("Loading FAISS index...")
        self.index = faiss.read_index(os.path.join(vectordb_path, "faiss_index.idx"))

        print("Loading document chunks metadata...")
        with open(os.path.join(vectordb_path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

        print(f"Ready with {len(self.chunks)} chunks.")

        print("Setting up Hugging Face Inference Client for Meta-LLaMA-3-8B-Instruct...")
        self.client = InferenceClient(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            provider="featherless-ai",
            token=os.getenv("HF_TOKEN")
        )

    def search(self, query: str, top_k: int = 3):
        """
        Performs semantic similarity search to retrieve top_k relevant chunks using FAISS.
        """
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        top_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return top_chunks

    def build_prompt(self, query: str, chunks: list):
        """
        Builds the prompt for the language model using retrieved chunks as context.
        """
        context = ""
        current_tokens = 0
        chunk_tokens = []

        for chunk in chunks:
            tokens = len(chunk.split())
            if current_tokens + tokens > 1500:
                break
            context += f"\n\n{chunk}"
            chunk_tokens.append(chunk)
            current_tokens += tokens

        messages = [
            {"role": "system", "content": (
                "You are a smart and helpful AI assistant trained to answer questions using the provided context. "
                "Your goal is to respond naturally and clearly, as if having a conversation. "
                "Use the context below to answer the user's question, but do not mention the context explicitly. "
                "Avoid saying things like 'based on the context' or 'the document says'. "
                "Instead, provide helpful, confident answers in your own words. "
                "If the answer involves steps or key points, use bullet points or a numbered list for clarity. "
                "If the context does not contain enough information, simply say: 'I don't know.'"
            )},
            {"role": "user", "content": f"Context:\n{context.strip()}\n\nQuestion: {query.strip()}"}
        ]


        return messages, chunk_tokens

    def generate_answer(self, messages):
        """
        Uses the LLaMA-3 model via the inference API to generate an answer.
        """
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                max_tokens=512,
                temperature=0.8,
                top_p=0.9
            )
            answer = completion.choices[0].message["content"].strip()
            return answer

        except Exception as e:
            return f"[ERROR] Could not generate answer: {str(e)}"
    
    def stream_answer(self, messages):
        """
        Streams the response from the LLaMA-3 model token by token.
        """
        try:
        
            stream = self.client.chat.completions.create(
                messages=messages,
                max_tokens=512,
                temperature=0.8,
                top_p=0.9,
                stream=True  
            )
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, "content"):
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[ERROR] Could not stream response: {str(e)}"

