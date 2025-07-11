import fitz  
import os

pdf_path = "../data/AI_Training_Document.pdf"
doc = fitz.open(pdf_path)

all_text = ""
for page in doc:
    text = page.get_text()
    
    # Filter out likely headers/footers
    lines = text.split("\n")
    lines = [line for line in lines if not (
        line.strip().lower().startswith("ebay") or 
        line.strip().lower().startswith("page") or 
        line.strip() == ''
    )]
    
    cleaned_text = "\n".join(lines)
    all_text += cleaned_text + "\n"

doc.close()

output_path = "../data/cleaned_text.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(all_text)

print("Cleaned text saved to 'cleaned_text.txt'")
