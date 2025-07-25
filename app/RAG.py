import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict

load_dotenv()

EMBED_MODEL = "l3cube-pune/bengali-sentence-similarity-sbert"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)
prompt = ChatPromptTemplate.from_template(
    """
কন্টেক্সট বিশ্লেষণ করে সংক্ষেপে প্রশ্নের উত্তর দাও।
###কন্টেক্সটঃ
{context}

###প্রশ্ন: {input}
###উত্তর:
""")

def build_vectorstore():
    file_path = "data/processed.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_text(text)
    vectorstore = Chroma.from_texts(splits, embedding=embedding, persist_directory="embeddings/chroma_store")
    vectorstore.persist()

if not os.path.exists("embeddings/chroma_store"): build_vectorstore()
vector_store = Chroma(persist_directory="embeddings/chroma_store", embedding_function=embedding)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State) -> Dict:
    docs = vector_store.similarity_search(state["question"])
    return {"context": docs}

def generate(state: State) -> Dict:
    context_str = "\n\n".join(doc.page_content for doc in state["context"])
    print(context_str)
    messages = prompt.invoke({"input": state["question"], "context": context_str})
    response = llm.invoke(messages)
    return {"answer": response.content}

rag_graph = StateGraph(State)
rag_graph.add_node("retrieve", retrieve)
rag_graph.add_node("generate", generate)
rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", "generate")

app_chain = rag_graph.compile()

def load_rag_chain():
    return app_chain
