import os
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain_core.documents import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define the request model
class QueryRequest(BaseModel):
    question: str

# Set up PostgreSQL connection
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Fetch data from PostgreSQL
def fetch_data_from_db():
    documents = []
    session = SessionLocal()
    try:
        result = session.execute(text("SELECT * FROM cve"))
        for row in result:
            # Assuming the data is stored as JSON in a column called 'data'
            documents.append(json.loads(row['data']))
    finally:
        session.close()
    return documents

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    chunks = []
    for doc in docs:
        for key, value in doc.items():
            if isinstance(value, dict):
                nested_chunks = chunk_data([value], chunk_size, chunk_overlap)
                chunks.extend(nested_chunks)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        nested_chunks = chunk_data([item], chunk_size, chunk_overlap)
                        chunks.extend(nested_chunks)
                    elif isinstance(item, str):
                        chunks.extend(split_text(item, chunk_size, chunk_overlap))
            elif isinstance(value, str):
                chunks.extend(split_text(value, chunk_size, chunk_overlap))
            else:
                chunks.append({key: value})
    return chunks

def split_text(text, chunk_size, chunk_overlap):
    chunked_text = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunked_text.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunked_text

def embed_query_with_retry(embeddings, query, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            return embeddings.embed_query(query)
        except Exception as e:
            if "RateLimitError" in str(e) or "insufficient_quota" in str(e):
                retries += 1
                wait_time = 2 ** retries
                print(f"Rate limit exceeded or insufficient quota. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Failed to embed query after several retries due to rate limit or quota issues.")

# Initialize Pinecone and OpenAI settings
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = 'vectordb'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

docs = fetch_data_from_db()
chunked_documents = chunk_data(docs)

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
documents = [Document(page_content=str(chunk), metadata={}) for chunk in chunked_documents]

# Create Pinecone vector store
pinecone_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=pinecone_store.as_retriever()
)

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        answer = qa.invoke(request.question).get("result")
        if not answer:
            raise HTTPException(status_code=404, detail="Answer not found")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
