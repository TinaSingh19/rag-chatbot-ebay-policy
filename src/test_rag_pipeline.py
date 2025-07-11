from rag_pipeline import RAGPipeline

# Initialize the RAG pipeline
rag = RAGPipeline(vectordb_path="../vectordb")

print("\nRAG Chatbot is ready. Type your question (or type 'exit' to quit):\n")

while True:
    user_input = input("User Question: ").strip()

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break

    if not user_input:
        continue

    # Step 1: Retrieve relevant chunks
    top_chunks = rag.search(user_input)

    # Step 2: Build prompt from retrieved chunks
    prompt_messages, used_chunks = rag.build_prompt(user_input, top_chunks)

    # Step 3: Generate answer from model
    answer = rag.generate_answer(prompt_messages)

    # Step 4: Display output
    print(f"\nBot Response: {answer}\n")

    # Step 5: Display source chunks
    print("Sources:")
    if used_chunks:
        for i, chunk in enumerate(used_chunks, 1):
            snippet = " ".join(chunk.strip().split())
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            print(f"{i}. {snippet}")
    else:
        print("No source chunks were used.")

    print()
