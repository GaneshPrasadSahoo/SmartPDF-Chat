import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.llms import Ollama
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate vector store from text chunks
def get_vector_store(text_chunks):
    # Using free HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.success("✅ Vector store created successfully!")

# Function to get conversational chain for PDF Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say "answer is not available in the context" and don't provide a wrong answer.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    # Using Ollama with your available models
    try:
        model = Ollama(model=st.session_state.selected_model)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error loading Ollama model: {e}")
        return None

# Function to process user question from PDFs
def chat_with_pdf(user_question):
    try:
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load FAISS vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        if chain:
            with st.spinner(f"🤔 Thinking with {st.session_state.selected_model}..."):
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("📄 **PDF Response:** ", response["output_text"])
        else:
            st.error("Failed to load the model chain")
            
    except Exception as e:
        st.error(f"Error in PDF chat: {e}")

# Function to chat with AI (general queries) using Ollama
def chat_with_ai(user_query):
    try:
        # Using Ollama for general chat
        model = Ollama(model=st.session_state.selected_model)
        with st.spinner(f"🤔 Thinking with {st.session_state.selected_model}..."):
            response = model.invoke(user_query)
        st.write("🤖 **AI Response:** ", response)
    except Exception as e:
        st.error(f"Error in AI chat: {e}")
        st.info("Make sure Ollama is running: Open terminal and run 'ollama serve'")

# Check if Ollama is running
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        return response.status_code == 200
    except:
        return False

# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with PDF & AI", page_icon="💁")
    st.header("SmartPDF Chat💁")

    # Initialize session state for model selection
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "llama3:latest"

    # Check Ollama status
    if not check_ollama_status():
        st.warning("⚠️ Ollama is not running. Please start Ollama first.")
        st.info("""
        **To start Ollama:**
        1. Open Terminal/Command Prompt
        2. Run: `ollama serve`
        3. Keep the terminal window open
        4. Refresh this page once Ollama is running
        """)
        return

    # Model selection in sidebar
    with st.sidebar:
        st.title("Menu:")
        
        # Model selection
        st.subheader("🤖 Select AI Model")
        model_option = st.selectbox(
            "Choose your model:",
            ["llama3:latest", "deepseek-r1:7b"],
            index=0 if st.session_state.selected_model == "llama3:latest" else 1
        )
        
        if model_option != st.session_state.selected_model:
            st.session_state.selected_model = model_option
            st.rerun()
        
        st.write(f"**Selected Model:** {st.session_state.selected_model}")
        
        # PDF processing section
        st.subheader("📄 PDF Processing")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
        
        if st.button("Submit & Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                    else:
                        st.error("No text could be extracted from the PDFs. Please try different PDF files.")
            else:
                st.warning("Please upload at least one PDF file.")

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["📄 Chat with PDF", "🤖 Chat with AI"])

    with tab1:
        st.subheader("Ask Questions from Your PDFs")
        st.info("💡 First upload and process PDFs using the sidebar, then ask questions here!")
        
        user_question = st.text_input("Enter your question based on uploaded PDFs:")
        if user_question:
            # Check if PDFs have been processed
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                chat_with_pdf(user_question)
            except:
                st.error("❌ No PDFs processed yet. Please upload and process PDFs first using the sidebar.")

    with tab2:
        st.subheader("Chat with AI 🤖")
        st.info("💡 This is for general conversations without PDF context")
        
        ai_question = st.text_input("Ask anything to AI:")
        if ai_question:
            chat_with_ai(ai_question)

    # Display model information in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("ℹ️ Model Information")
        if st.session_state.selected_model == "llama3:latest":
            st.write("**Llama 3** - General purpose model, good for most tasks")
        else:
            st.write("**DeepSeek R1** - Specialized in reasoning and coding")

if __name__ == "__main__":
    main()