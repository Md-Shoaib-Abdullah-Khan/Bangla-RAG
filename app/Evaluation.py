import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from RAG import load_rag_chain

embedding_model = HuggingFaceEmbeddings(model_name="l3cube-pune/bengali-sentence-similarity-sbert")

def load_qa_pairs(file_path: str) -> list[dict]:
    """Load questions and reference answers from text file"""
    qa_pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                question, answer = line.strip().split('|', 1)
                qa_pairs.append({
                    "question": question,
                    "reference_answer": answer
                })
            except ValueError:
                continue
    return qa_pairs

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts"""
    emb1 = embedding_model.embed_query(text1)
    emb2 = embedding_model.embed_query(text2)
    return cosine_similarity([emb1], [emb2])[0][0]

def evaluate_rag(rag_chain, qa_pairs: list[dict]) -> list[dict]:
    """Evaluate RAG pipeline against test cases"""
    results = []
    
    for pair in qa_pairs:
        try:
            session_id = 'Default'
            rag_result = rag_chain.invoke(
            {"question": pair["question"]},
            config={
                "configurable": {
                    "thread_id": session_id, 
                    "session_id": session_id  
                }
            }
        )
 
            generated_answer = rag_result["answer"]
            
            similarity = calculate_similarity(
                generated_answer,
                pair["reference_answer"]
            )
            
            results.append({
                "question": pair["question"],
                "reference_answer": pair["reference_answer"],
                "generated_answer": generated_answer,
                "similarity_score": similarity,
                "contexts": [doc.page_content for doc in rag_result["context"]]
            })
            
        except Exception as e:
            print(f"Error processing question: {pair['question']}\nError: {str(e)}")
            results.append({
                "question": pair["question"],
                "error": str(e)
            })
    
    return results

def print_summary(results: list[dict]):
    """Print evaluation results for each query"""
    print("\nRAG Evaluation Results:")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}:")
        print(f"Question: {result['question']}")
        
        if "error" in result:
            print(f"Error: {result['error']}")
            continue
            
        print(f"Reference Answer: {result['reference_answer']}")
        print(f"Generated Answer: {result['generated_answer']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print("-" * 50)
    
    successful_runs = [r for r in results if "similarity_score" in r]
    if successful_runs:
        avg_similarity = np.mean([r["similarity_score"] for r in successful_runs])
        print(f"\nAverage Similarity Score: {avg_similarity:.4f}")
    else:
        print("\nNo successful evaluations to calculate average similarity.")

if __name__ == "__main__":
    
    rag_chain = load_rag_chain() 

    qa_pairs = load_qa_pairs("app/test_cases.txt")
    #print(qa_pairs)
 
    evaluation_results = evaluate_rag(rag_chain, qa_pairs)
    
    print_summary(evaluation_results)