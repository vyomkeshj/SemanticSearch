from fastapi import FastAPI

from search_engine import SearchEngine

app = FastAPI()
search_engine = SearchEngine()


@app.get("/search")
def search_in_corpus_embeddings(queries: str):
    search_engine.search_in_corpus_embeddings(queries)
    return {"queries": queries}
