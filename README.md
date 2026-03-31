# Step 6 - Relevant Chunk Retriever in RAG 

<img width="1536" height="1024" alt="Thumbnail" src="https://github.com/user-attachments/assets/f75e210c-c78e-4a2f-8575-d4cf66837a78" />

# YouTube video with detailed explaination and demo https://youtu.be/-qJwyLAuzTY

Finds and returns the most relevant chunks from the database based on query similarity

It is the bridge between your question and the data

# How it works?
Vector Search: It converts your question into a mathematical representation (an embedding)

Similarity Match: It scans a database of pre-processed documents to find text chunks that are mathematically closest to your question

The Hand-off: It grabs the top few results (the "relevant chunks") and hands them to the LLM (e.g. GPT-4)

