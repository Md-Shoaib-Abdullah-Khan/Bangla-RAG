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
                    "thread_id": session_id,  # Use session_id as thread_id
                    "session_id": session_id  # Keep for your own tracking
                }
            }
        )
    return {
        "question": input.query,
        "answer": result["answer"]
    }

# @app.get("/status")
# def status():
#     return status_check()

# ----- MAIN RUNNER -----
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
