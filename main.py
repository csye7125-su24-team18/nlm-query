import os
import json
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define the request model
class QueryRequest(BaseModel):
    question: str

def read_json_files(directory):
    documents = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        documents.append(data)
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON in file: {file_path}")
    return documents

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    chunks = []
    for doc in docs:
        doc_str = json.dumps(doc)  # Convert the JSON document to a string
        split_docs = split_text(doc_str, chunk_size, chunk_overlap)
        chunks.extend(split_docs)
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
                logger.warning(f"Rate limit exceeded or insufficient quota. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Failed to embed query after several retries due to rate limit or quota issues.")

def initialize_pinecone():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = 'vectordb'
    if index_name not in pc.list_indexes().names():
        logger.info(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # Adjust dimension according to the embedding model used
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        time.sleep(60)  # Wait for the index to be fully created
    else:
        logger.info(f"Index {index_name} already exists")
    return pc, index_name

# Initialize Pinecone and load documents
pc, index_name = initialize_pinecone()

# Read and chunk documents
docs = read_json_files('cve')
chunked_documents = chunk_data(docs)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

documents = [Document(page_content=chunk) for chunk in chunked_documents]

# Create Pinecone vector store
pinecone_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Initialize HuggingFacePipeline for LLM
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
hf_pipeline = pipeline(
    "text2text-generation", model=model, tokenizer=tokenizer, max_length=100
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Set up the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=pinecone_store.as_retriever(search_kwargs={"k": 3})
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this to match your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        logger.info(f"Received question: {request.question}")
        prompt = f"Answer the following question based on the given context: {request.question}"
        logger.info(f"Generated prompt: {prompt}")
        result = qa({"query": request.question})
        answer = result.get("result", "").strip()
        logger.info(f"Generated answer: {answer}")
        if not answer:
            raise HTTPException(status_code=404, detail="Answer not found")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
