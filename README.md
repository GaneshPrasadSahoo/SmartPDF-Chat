# 💁 SmartPDF Chat

**SmartPDF Chat** is an AI-powered application built with **Streamlit**, **LangChain**, and **Ollama**, allowing users to **chat with their PDF documents** or have general **AI conversations** using locally hosted LLMs (like **Llama 3** or **DeepSeek R1**).  

It intelligently extracts text from PDFs, creates vector embeddings for semantic understanding, and answers user questions accurately based on the uploaded content.

---

## 🚀 Features

✅ **Chat with PDFs** — Upload any PDF (NCERT, research papers, notes, etc.) and ask questions about the content.  
✅ **General AI Chat** — Talk with powerful AI models like *Llama 3* or *DeepSeek R1* for general queries.  
✅ **Offline/Local AI** — Uses **Ollama** to run LLMs locally (no OpenAI API key required).  
✅ **Smart Embeddings** — Uses **HuggingFace Sentence Transformers** and **FAISS** for efficient semantic search.  
✅ **Streamlit UI** — Clean and interactive interface for effortless use.  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend/UI** | Streamlit |
| **PDF Parsing** | PyPDF2 |
| **AI Models** | Ollama (Llama 3, DeepSeek R1) |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS |
| **Environment Variables** | python-dotenv 

---

## ⚙️ Installation Guide

Follow these steps to run SmartPDF Chat locally 👇  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/SmartPDF-Chat.git
cd SmartPDF-Chat

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Start Ollama Server
Make sure Ollama is installed and running locally.
To start Ollama, open a new terminal and run:
ollama serve

Then, pull the required models:
ollama pull llama3
ollama pull deepseek-r1:7b

5️⃣ Run the Application
streamlit run app.py


📄 Usage

1.Upload PDFs from the sidebar under 📄 PDF Processing.
2.Click Submit & Process PDFs to create vector embeddings.
3.Go to the 📄 Chat with PDF tab and start asking questions related to your uploaded documents.
4.Use the 🤖 Chat with AI tab for general chat or coding-related questions.
5.You can switch between Llama 3 and DeepSeek R1 in the sidebar model selector.


🧑‍💻 Example Questions
📄 Chat with PDF:

“What are the main topics discussed in Chapter 5 of this biology textbook?”
“Summarize the abstract of this research paper.”
“Who are the authors mentioned in this paper?”

🤖 Chat with AI:

“Explain overfitting in machine learning.”
“Write a Python function to calculate factorial.”
“What’s new in Llama 3 compared to GPT-4?”

🔒 Notes
 Ollama must be running locally for the app to work.
 Make sure to process PDFs before asking questions.
 If no text is extracted, try uploading a text-based PDF (not a scanned image).

💡 Future Improvements
 Add OCR support for scanned PDFs using Tesseract
 Include conversation memory for contextual chat
 Support multiple document uploads and cross-document Q&A
 Add UI themes and chat history saving

👨‍💻 Author
Ganesh Prasad Sahoo
💼 B.Tech CSE | Centurion University of Technology and Management
🌐 LinkedIn : https://www.linkedin.com/in/ganesh-prasad-sahoo-775346293/
