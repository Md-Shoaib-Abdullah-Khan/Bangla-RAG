from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
EMBED_MODEL = "l3cube-pune/bengali-sentence-similarity-sbert"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

prompt = ChatPromptTemplate([(
    """
কন্টেক্সট বিশ্লেষণ করে সংক্ষেপে প্রশ্নের উত্তর দাও।
কন্টেক্সটঃ
{context}

প্রশ্ন: {input}
উত্তর:
""")
])

def build_vectorstore():
    file_path = "data/processed.txt" 

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            docs = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=200,  
        add_start_index=True, 
    )
    hf = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    all_splits = text_splitter.split_text(docs)
    vector = Chroma.from_texts(
        texts = all_splits, 
        embedding = hf,
        persist_directory = "embeddings/chroma_store"
    )

    
    
def load_rag_chain():
    vectorstore = Chroma(
        persist_directory="embeddings/chroma_store",
        embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    print(retriever.invoke("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"))
    
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=groq_api_key)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain


if not os.path.exists("embeddings/chroma_store"): build_vectorstore()

build_vectorstore()

question = "বিয়ের সময় মেয়ের প্রকৃত বয়স কত ছিল?"

rag_chain = load_rag_chain()

response = rag_chain.invoke({'input' : question})
print(response['answer'])