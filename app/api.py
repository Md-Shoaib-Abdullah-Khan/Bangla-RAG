from fastapi import FastAPI
from RAG import load_rag_chain
from pydantic import BaseModel
import uvicorn

app = FastAPI()
rag_chain = load_rag_chain()

class QueryInput(BaseModel):
    query: str


@app.post("/ask")
async def ask(input: QueryInput):
    return rag_chain.invoke({'input':input.query})['answer'].split('>')[-1]


# ----- MAIN RUNNER -----
if __name__ == "__main__":
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
