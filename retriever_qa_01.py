import subprocess
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

text_data = """
Retriever: In a RAG system, the retriever is responsible for finding the most relevant documents from a database based on a user's query.
Generator: The generator takes the retrieved documents and the original query to produce a final, coherent response.
Ollama: A tool that allows you to run large language models locally.
Recent studies in Oncology suggest that wearable monitoring improves outcomes.
The standard protocol for Type 2 Diabetes involves metformin.
The side effects of Lisinopril include nausea.
Patient X presented with chest pain and was diagnosed with Atrial Fibrillation.
To prevent Hypertension, patients are advised to reduce sodium.
Insurance coverage for MRI scans typically requires clinical necessity.
Recent studies in Mental Health suggest that personalized medicine improves outcomes.
The standard protocol for Asthma involves beta-blockers.
Patient X presented with fatigue and was diagnosed with Rheumatoid Arthritis.
To prevent Eczema, patients are advised to increase fiber intake.
Insurance coverage for hip replacement typically requires prior authorization.
Recent studies in Telemedicine suggest that AI-driven diagnostics improves outcomes.
The side effects of Amoxicillin include dry mouth.
Patient X presented with blurred vision and was diagnosed with Type 2 Diabetes.
To prevent Atrial Fibrillation, patients are advised to stay hydrated.
The standard protocol for Hypertension involves insulin therapy.
Insurance coverage for blood tests typically requires a referral from a GP.
Recent studies in Pediatrics suggest that gene therapy improves outcomes.
The side effects of Atorvastatin include dizziness.
Patient X presented with shortness of breath and was diagnosed with Asthma.
The side effects of Sertraline include dry mouth.
Recent studies in Orthopedics suggest that personalized medicine improves outcomes.
To prevent Type 2 Diabetes, patients are advised to exercise 30 mins daily.
Insurance coverage for angioplasty typically requires prior authorization.
Patient X presented with persistent cough and was diagnosed with Eczema.
The standard protocol for Atrial Fibrillation involves chemotherapy.
Recent studies in Nutrition suggest that wearable monitoring improves outcomes.
To prevent Hypertension, patients are advised to improve sleep hygiene.
The side effects of Albuterol include fatigue.
Insurance coverage for vaccinations typically requires clinical necessity.
"""

# 1. Start ollama programmatically
try:
    result = subprocess.run(['ollama', 'list'], capture_output=False, text=False, check=False)
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
query_list = [
    "What are the common side effects of Lisinopril?",
    "How should a patient manage Type 2 Diabetes through lifestyle?",
    "What is the protocol for treating Atrial Fibrillation?",
    "Does insurance usually cover hip replacement surgery?",
    "What symptoms are associated with Rheumatoid Arthritis in patients?",
    "How does AI-driven diagnostics affect Cardiology outcomes?",
    "What are the requirements for an MRI scan authorization?",
    "What advice is given to patients to prevent Asthma attacks?",
    "What is the standard treatment for Hypertension?",
    "Which medication causes dizziness and nausea as side effects?",
    "How does personalized medicine improve Pediatrics?",
    "What are the benefits of early screening in Oncology?",
    "What lifestyle changes help with Eczema?",
    "What is involved in the standard protocol for Asthma?",
    "How long should a patient fast before blood tests?",

    # "How do I reset my Windows password?",
    # "What is the best recipe for chocolate chip cookies?",
    # "Who won the FIFA World Cup in 2022?",
    # "How do black holes form in space?",
    # "What are the top-rated hiking trails in the Himalayas",
]

for qry in query_list:
    relevant_chunks = retriever.invoke(qry)
    # Output results
    print(f"Query: {qry}")
    for i, chunk in enumerate(relevant_chunks):
        print(f"\nResult {i+1}:")
        print(chunk.page_content)
    print("===" * 10)
