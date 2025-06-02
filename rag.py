import os
import google.generativeai as genai
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify

from vector_store import VectorStore

app = Flask(__name__)

GEMINI_API_KEY = "API_KEY"
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please set it as an environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

vector_store = VectorStore(collection_name="medical_documents")

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip()

def generate_response(prompt, use_rag=True):
    prohibited_keywords = ["medicine", "prescription", "treatment", "dose", "drug", "take for"]
    if any(keyword in prompt.lower() for keyword in prohibited_keywords):
        return "I'm sorry, but I can't provide medical advice or prescriptions. Please consult a healthcare professional."

    context = ""
    if use_rag:
        results = vector_store.query(prompt)
        if results["documents"] and results["documents"][0]:
            context_docs = results["documents"][0]
            context = "\n\n".join(f"Document context: {doc}" for doc in context_docs)

    if context:
        enhanced_prompt = f"""
        I'm going to answer a question based on the following context information:
        
        {context}
        
        Question: {prompt}
        
        Answer:
        """
    else:
        enhanced_prompt = prompt

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(enhanced_prompt)
    
    return {
        "response": response.text,
        "rag_used": bool(context),
        "documents_used": results["documents"][0] if use_rag and context else []
    }

@app.route("/")
def home():
    return "RAG API is running. Use /extract_text, /generate_response, or /documents endpoints."

@app.route("/extract_text", methods=["POST"])
def extract_text_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        text = extract_text_from_image(image_file)
        return jsonify({"extracted_text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_response", methods=["POST"])
def generate_response_endpoint():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    use_rag = data.get('use_rag', True)
    
    try:
        response_data = generate_response(data['prompt'], use_rag=use_rag)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/add_document", methods=["POST"])
def add_document_endpoint():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No document text provided"}), 400
    
    try:
        title = data.get('title', 'Untitled Document')
        metadata = data.get('metadata', {})
        metadata['title'] = title
        
        doc_id = vector_store.add_document(data['text'], metadata)
        
        return jsonify({"success": True, "document_id": doc_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/documents", methods=["GET"])
def list_documents_endpoint():
    try:
        documents = vector_store.get_all_documents()
        return jsonify({"documents": documents})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document_endpoint(doc_id):
    try:
        vector_store.delete_document(doc_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5081, host="0.0.0.0", debug=True)
