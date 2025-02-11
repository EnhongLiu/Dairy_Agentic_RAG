import os
from langchain.vectorstores import FAISS, Chroma
from langchain.schema import Document
from langchain.embeddings import DatabricksEmbeddings
from langchain.chat_models import ChatDatabricks


## Build vector store from chunks. Still, if chroma db realdy exists, just skip to load
# Embedding Model
os.environ["DATABRICKS_TOKEN"]= 'xxxxx'
embd = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# Chat Model
llama70b = ChatDatabricks(
    endpoint="databricks-meta-llama-3-1-70b-instruct",
    temperature=0.0,
    max_tokens=1024,
)

# Build Vector Store
persist_dir='xxxx'
os.makedirs(persist_dir, exist_ok=True)

formatted_chunks = [Document(page_content = chunk.text, metadata = {"source": chunk.meta.origin.filename}) for chunk in chunk_iter]


vectorstore_from_documents = FAISS.from_documents(
    documents= formatted_chunks,
    embedding= embd,
)

vectorstore_from_documents.save_local(persist_dir)

vectorstore_from_documents = Chroma.from_documents(
    documents= formatted_chunks,
    collection_name="rag-chroma-google-meta-apple-microsoft",
    embedding=embd,
    persist_directory= persist_dir
)
