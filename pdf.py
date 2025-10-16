import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("DEEP_SEEK")

# Configure Google Generative AI with API key
genai.configure(api_key=api_key)

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
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")  # ✅ Correct model name
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load the conversational chain for PDF Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say 
    "answer is not available in the context" and don't provide a wrong answer.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user question from PDFs
def chat_with_pdf(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")  # ✅ Correct model name
    
    # Load FAISS vector store safely
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("📄 **PDF Response:** ", response["output_text"])

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDF - Ganesh GPT", page_icon="📘", layout="wide")
    st.title("📘 Chat with your PDF using Gemini 🔍")

    pdf_docs = st.file_uploader("Upload your PDF files here", accept_multiple_files=True)
    
    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Extracting and processing text..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ PDF processed successfully! You can now ask questions.")
        else:
            st.warning("⚠️ Please upload at least one PDF file before processing.")
    
    user_question = st.text_input("💬 Ask a question about your PDF:")
    
    if user_question:
        with st.spinner("Generating answer..."):
            chat_with_pdf(user_question)

if __name__ == "__main__":
    main()
