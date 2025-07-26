from fastapi import FastAPI, Query
from RAG import load_rag_chain
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class QueryInput(BaseModel):
    query: str

rag_graph = load_rag_chain()

@app.post("/ask")
async def ask(input: QueryInput, session_id: str = Query(default="default")):
    result = rag_graph.invoke(
            {"question": input.query},
            config={
                "configurable": {
                    "thread_id": session_id, 
                    "session_id": session_id  
                }
            }
        )
    return {
        "question": input.query,
        "answer": result["answer"]
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
