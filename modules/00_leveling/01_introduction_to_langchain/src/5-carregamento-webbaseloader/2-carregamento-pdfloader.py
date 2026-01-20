from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader('./5-carregamento-webbaseloader/gpt-5-system-card.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks = splitter.split_documents(docs)

for chunk in chunks:
    print(chunk)
    print('-'*30)