# 🇧🇩 Bilingual RAG Chatbot (Bangla + English)

This project implements a **Retrieval-Augmented Generation (RAG)** based chatbot that can understand and answer both **Bangla** and **English** queries. It retrieves context from a PDF knowledge base and uses a Language Model to generate grounded and relevant answers.

---

## 📁 Project Structure

.
├── app/
│ ├── api.py # FastAPI logic for querying RAG
│ ├── Evaluation.py # RAG performance evaluator using cosine similarity
│ ├── main.py # Streamlit chatbot UI
│ ├── RAG.py # RAG pipeline using LangGraph
│ └── test_cases.txt # Predefined test queries with expected answers
├── data/
│ ├── bangla.pdf # Raw Bangla textbook input
│ └── processed.txt # Cleaned and chunked version of the textbook
├── Preprocess.ipynb # Jupyter Notebook for extracting and cleaning text from PDF
├── requirements.txt # Python dependencies
└── README.md # 📍 You're here


---

## 🚀 Features

- ✅ Accepts user queries in **Bangla and English**
- ✅ Retrieves relevant document chunks using vector search
- ✅ Generates accurate answers using `ChatGroq` (LLM)
- ✅ Memory-enabled: maintains conversation history
- ✅ Built-in **RAG evaluation**: measures groundedness & relevance
- ✅ REST API for integration
- ✅ Streamlit chatbot interface

---

## 🔧 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/bilingual-rag-chatbot.git
cd bilingual-rag-chatbot

2. Create & activate virtual environment

python -m venv myenv
source myenv/bin/activate        # On Windows: myenv\\Scripts\\activate

3. Install dependencies

pip install -r requirements.txt

4. Create .env file

GROQ_API_KEY=your_groq_api_key

5. Preprocess the Bangla PDF

Open and run the Jupyter notebook:

jupyter notebook Preprocess.ipynb

This will extract the Bangla content from data/bangla.pdf and save it as processed.txt.
▶️ Running the System
✅ Run the FastAPI server

uvicorn app.api:app --reload

    API Endpoint: http://localhost:8000/ask?session_id=your-session

    Accepts JSON like:

{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}

✅ Run the Streamlit Chatbot

streamlit run app/main.py

📊 Evaluation

Run the evaluation script to test RAG performance:

python app/Evaluation.py

This script compares the RAG-generated answers to expected answers in test_cases.txt and calculates:

    🔹 Groundedness Score: similarity between answer and retrieved context

    🔹 Relevance Score: similarity between query and retrieved context

🧪 Example Test Case

Input: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
Expected: ১৫ বছর
RAG Output: ✅ Retrieved from vector store
Answer: ১৫ বছর
Groundedness Score: 0.89
Relevance Score: 0.93
🤖 Tech Stack

    LangChain

    LangGraph

    ChromaDB

    FastAPI

    Streamlit

    GROQ for LLM

    Sentence Transformers for semantic similarity

✍️ Author

Shoaib Khan
An AI enthusiast exploring multilingual education tools.
📄 License

This project is licensed under the MIT License.


---
