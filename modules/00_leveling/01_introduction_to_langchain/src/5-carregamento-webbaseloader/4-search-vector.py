import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

_ = load_dotenv()

query = "Tell me more about the gpt-5 thinking evaluation and performance results comparing to gpt-4"


embeddings = OpenAIEmbeddings(model = os.getenv('OPENAI_MODEL', 'text-embedding-3-small'))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv('PGVECTOR_COLLECTION'),
    connection=os.getenv('PGVECTOR_URL'),
    use_jsonb=True,
)

results = store.similarity_search_with_score(query, k=3)

# 1. Create a retriever from your vector store
retriever = store.as_retriever(search_kwargs={"k": 4})

# 2. Define the Language Model you want to use
llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0, verbose=True)

# 3. Create the QA chain
# This chain automatically fetches docs and passes them to the LLM
qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff", # 'stuff' combines all docs into one prompt
    retriever=retriever,
    return_source_documents=True # Optional: returns the docs used
)

# 4. Run the chain with your original question
result = qa_chain.invoke({"query": query})

# The 'result' will contain the final AI-generated answer.
print(result)