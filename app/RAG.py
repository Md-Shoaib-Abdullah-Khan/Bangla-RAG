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
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated
from langdetect import detect
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

EMBED_MODEL = "l3cube-pune/bengali-sentence-similarity-sbert"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)  
translation_llm = ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)  


translate_to_bn_prompt = ChatPromptTemplate.from_template(
    "Translate this English text to Bangla: {text}"
)

translate_to_en_prompt = ChatPromptTemplate.from_template(
    "Translate this Bangla text to English: {text}"
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """
প্রাসঙ্গিক তথ্য ও চ্যাটের ইতিহাস বিশ্লেষণ করে প্রশ্নের সংক্ষিপ্ত উত্তর দাও।
### চ্যাটের ইতিহাস:
{history}

### প্রাসঙ্গিক তথ্য:
{context}
"""),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}")
])


def detect_language(text: str) -> str:
    """Detect language of input text (en/bn)"""
    try:
        lang = detect(text)
        return 'bn' if lang == 'bn' else 'en'
    except:
        return 'en'

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text between English and Bangla"""
    if source_lang == target_lang:
        return text
        
    prompt = translate_to_bn_prompt if target_lang == 'bn' else translate_to_en_prompt
    messages = prompt.invoke({"text": text})
    response = translation_llm.invoke(messages)
    return response.content

def build_vectorstore():
    """Build vector store from documents"""
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

if not os.path.exists("embeddings/chroma_store"): 
    build_vectorstore()
vector_store = Chroma(persist_directory="embeddings/chroma_store", embedding_function=embedding)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_language: str 

def retrieve(state: State) -> Dict:
    """Retrieve relevant documents"""
    question = state["question"]
    lang = detect_language(question)
    
    if lang == 'en':
        translated_question =  translate_text(question, 'en', 'bn')
        docs = vector_store.similarity_search(translated_question)
    else:
        docs = vector_store.similarity_search(question)
    print(docs)
    return {
        "context": docs,
        "original_language": lang
    }

def generate(state: State) -> Dict:
    """Generate response with translation handling"""
    
    history = "\n".join(
        f"ব্যবহারকারী: {msg.content}" if isinstance(msg, HumanMessage) 
        else f"সহকারী: {msg.content}" 
        for msg in state["messages"][-6:]
    )
    
    context_str = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.invoke({
        "input": state["question"],
        "context": context_str,
        "history": history,
        "messages": state["messages"][-6:]
    })
    response = llm.invoke(messages)
    answer = response.content.split('>')[-1].strip('\n')
    
    if state.get("original_language") == 'en':
        answer =  translate_text(answer, 'bn', 'en')
    
    return {
        "answer": answer,
        "original_language": state["original_language"]
    }

rag_graph = StateGraph(State)
rag_graph.add_node("retrieve", retrieve)
rag_graph.add_node("generate", generate)
rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", "generate")

memory = MemorySaver()
app_chain = rag_graph.compile(
    checkpointer=memory,
    interrupt_after=["generate"]
)

def load_rag_chain():
    return app_chain