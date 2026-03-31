import subprocess
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

text_data = """
Retriever: In a RAG system, the retriever is responsible for finding the most relevant documents from a database based on a user's query.
Generator: The generator takes the retrieved documents and the original query to produce a final, coherent response.
Ollama: A tool that allows you to run large language models locally.
"""

# 1. Start ollama programmatically
try:
    result = subprocess.run(['ollama', 'list'], capture_output=False, text=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error running 'ollama list': {e.stderr}")
except FileNotFoundError:
    print("Ollama is not installed or not in your system PATH.")

# 2. Split text into manageable chunks
text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20, separator="\n")
docs = [Document(page_content=x) for x in text_splitter.split_text(text_data)]

# 3. Initialize Local Embeddings using Ollama
# 'nomic-embed-text' is a highly efficient local embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Create a Local Vector Store (Chroma)
# This converts the text chunks into vectors and stores them in memory
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# 5. Define the Retriever
# 'k=1' tells the retriever to return only the single most relevant chunk
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# 6. Perform Retrieval
query = "How does a retriever work?"
relevant_chunks = retriever.invoke(query)

# Output results
print(f"Query: {query}")
for i, chunk in enumerate(relevant_chunks):
    print(f"\nResult {i+1}:")
    print(chunk.page_content)
