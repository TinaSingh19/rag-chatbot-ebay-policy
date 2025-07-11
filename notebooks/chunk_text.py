import os
import spacy

nlp = spacy.load("en_core_web_sm")

# Read cleaned text
with open(os.path.join("..", "data", "cleaned_text.txt"), "r", encoding="utf-8") as f:
    raw_text = f.read()

# Sentence aware splitting
doc = nlp(raw_text)
sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

# Chunking logic: Chunking into ~150-word chunks
chunks = []
current_chunk = ""
word_limit = 150

for sentence in sentences:
    if len(current_chunk.split()) + len(sentence.split()) <= word_limit:
        current_chunk += " " + sentence
    else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence

if current_chunk:
    chunks.append(current_chunk.strip())

output_path = os.path.join("..", "chunks", "document_chunks.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"### Chunk {i+1} ###\n{chunk}\n\n")

print(f"Created {len(chunks)} chunks and saved to 'chunks/document_chunks.txt'")