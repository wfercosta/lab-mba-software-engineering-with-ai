import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

_ = load_dotenv()

current_dir = Path(__file__).parent
file_path = str(current_dir / "gpt-5-system-card.pdf")

docs = PyPDFLoader(file_path).load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False)
splits = splitter.split_documents(docs)

if not splits:
    raise SystemExit(0)


documents = [
    Document(
        page_content=d.page_content, 
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    ) for d in splits
]

ids = [f'docs-{i}' for i in range(len(documents))]

embeddings = OpenAIEmbeddings(model = os.getenv('OPENAI_MODEL', 'text-embedding-3-small'))

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv('PGVECTOR_COLLECTION'),
    connection=os.getenv('PGVECTOR_URL'),
    use_jsonb=True,
)

store.add_documents(documents, ids=ids)