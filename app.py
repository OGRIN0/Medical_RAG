import streamlit as st
import requests
import io
from PIL import Image
import pdf2image
import tempfile
import os

API_BASE_URL = "http://localhost:5081"

st.set_page_config(page_title="RAG Medical Assistant", page_icon="🏥", layout="wide")

st.title("📋 RAG Medical Assistant")
st.markdown("Upload medical documents or ask questions - I'm here to help!")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "documents" not in st.session_state:
    st.session_state.documents = []

tab1, tab2, tab3 = st.tabs(["Chat with AI", "Extract Text from Image", "Document Management"])

with tab1:
    st.header("Chat with AI")
    
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message.get("role") == "assistant" and message.get("documents_used"):
                with st.expander("Documents used for this response"):
                    for i, doc in enumerate(message["documents_used"]):
                        st.markdown(f"**Document {i+1}:**\n{doc[:500]}...", unsafe_allow_html=False)
    
    use_rag = st.sidebar.checkbox("Use RAG for better responses", value=True)
    
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/generate_response",
                    json={"prompt": user_input, "use_rag": use_rag}
                )
                if response.status_code == 200:
                    data = response.json()
                    ai_response = data["response"]
                    documents_used = data.get("documents_used", [])
                    rag_used = data.get("rag_used", False)
                else:
                    ai_response = f"Error: {response.json().get('error', 'Unknown error')}"
                    documents_used = []
                    rag_used = False
            except requests.exceptions.ConnectionError:
                ai_response = "Error: Cannot connect to the backend server. Please make sure it's running."
                documents_used = []
                rag_used = False
        
        response_data = {
            "role": "assistant", 
            "content": ai_response,
            "documents_used": documents_used,
            "rag_used": rag_used
        }
        st.session_state.conversation.append(response_data)
        with st.chat_message("assistant"):
            st.write(ai_response)
            if documents_used:
                with st.expander("Documents used for this response"):
                    for i, doc in enumerate(documents_used):
                        st.markdown(f"**Document {i+1}:**\n{doc[:500]}...", unsafe_allow_html=False)

with tab2:
    st.header("Extract Text from Medical Documents")
    st.write("Upload an image or PDF of a medical document to extract its text.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])
    
    if uploaded_file:
        is_pdf = uploaded_file.type == "application/pdf"
        
        if is_pdf:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                images = pdf2image.convert_from_path(pdf_path, first_page=1, last_page=1)
                if images:
                    pdf_image = images[0]
                    st.image(pdf_image, caption="First page of uploaded PDF", use_column_width=True)
                    
                    os.unlink(pdf_path)
                else:
                    st.error("Could not extract any images from the PDF.")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.info("Make sure pdf2image and poppler are properly installed.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                try:
                    uploaded_file.seek(0)
                    
                    if is_pdf:
                        all_text = []
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            pdf_path = tmp_file.name
                        
                        images = pdf2image.convert_from_path(pdf_path)
                        
                        for i, img in enumerate(images):
                            with st.spinner(f"Processing page {i+1} of {len(images)}..."):
                                temp_img_path = f"{pdf_path}_page{i+1}.jpg"
                                img.save(temp_img_path)
                                
                                with open(temp_img_path, "rb") as img_file:
                                    files = {"image": img_file}
                                    response = requests.post(f"{API_BASE_URL}/extract_text", files=files)
                                
                                os.unlink(temp_img_path)
                                
                                if response.status_code == 200:
                                    page_text = response.json()["extracted_text"]
                                    all_text.append(f"--- Page {i+1} ---\n{page_text}")
                        
                        os.unlink(pdf_path)
                        
                        extracted_text = "\n\n".join(all_text)
                    else:
                        files = {"image": uploaded_file}
                        response = requests.post(f"{API_BASE_URL}/extract_text", files=files)
                        extracted_text = response.json()["extracted_text"] if response.status_code == 200 else f"Error: {response.json().get('error', 'Unknown error')}"
                    
                    st.text_area("Extracted Text", extracted_text, height=250)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Add to Knowledge Base"):
                            doc_title = st.text_input("Document Title", 
                                                     value=uploaded_file.name if uploaded_file.name else "Untitled")
                            
                            if st.button("Confirm Add to Knowledge Base"):
                                response = requests.post(
                                    f"{API_BASE_URL}/add_document",
                                    json={"text": extracted_text, "title": doc_title}
                                )
                                
                                if response.status_code == 200:
                                    st.success(f"Document '{doc_title}' added to knowledge base!")
                                    st.experimental_rerun()
                                else:
                                    st.error(f"Error adding document: {response.json().get('error', 'Unknown error')}")
                    
                    with col2:
                        if st.button("Ask about this document"):
                            st.session_state.conversation.append(
                                {"role": "system", "content": f"Document content: {extracted_text}"}
                            )
                            st.experimental_rerun()
                
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend server. Please make sure it's running.")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

with tab3:
    st.header("Document Management")
    st.write("View and manage documents in your knowledge base.")
    
    try:
        with st.spinner("Loading documents..."):
            response = requests.get(f"{API_BASE_URL}/documents")
            if response.status_code == 200:
                documents = response.json().get("documents", [])
                st.session_state.documents = documents
            else:
                st.error(f"Error fetching documents: {response.json().get('error', 'Unknown error')}")
                documents = []
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the backend server. Please make sure it's running.")
        documents = []
    
    if not documents:
        st.info("No documents in the knowledge base. Add documents from the 'Extract Text' tab.")
    else:
        st.write(f"Total documents: {len(documents)}")
        for doc in documents:
            with st.expander(f"{doc.get('metadata', {}).get('title', 'Untitled')}"):
                st.write(f"ID: {doc['id']}")
                st.write(f"Chunks: {doc['chunk_count']}")
                st.write("Sample text:")
                st.write(doc['sample_text'])
                
                if st.button(f"Delete Document", key=f"delete_{doc['id']}"):
                    with st.spinner("Deleting document..."):
                        response = requests.delete(f"{API_BASE_URL}/documents/{doc['id']}")
                        if response.status_code == 200:
                            st.success("Document deleted!")
                            st.experimental_rerun()
                        else:
                            st.error(f"Error deleting document: {response.json().get('error', 'Unknown error')}")

st.sidebar.header("About")
st.sidebar.info(
    "This application uses RAG (Retrieval-Augmented Generation) to help you with medical documents. "
    "However, please note that this tool does not provide medical advice. Always consult with "
    "healthcare professionals for medical guidance."
)

