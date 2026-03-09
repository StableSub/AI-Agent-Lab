from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

file_path = "/Users/anjeongseob/Desktop/Storage/AI/LangChain/RAG/ARK_INNOVATION_ETF_ARKK_HOLDINGS.pdf"

loader = PDFPlumberLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)

recursive_docs = text_splitter.split_documents(docs)

underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

store = LocalFileStore("./.cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)

vectorestore = InMemoryVectorStore.from_documents(
    recursive_docs,
    cached_embedder
)

query = "Tesla 투자 비중이 얼마나 돼?"
# results = vectorestore.similarity_search(query)

# print(f"검색된 문서 내용 {results[0].page_content}")

from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"

db = Chroma.from_documents(
    documents=recursive_docs,
    embedding=cached_embedder,
    persist_directory=CHROMA_PATH,
    collection_name="rag_collection"
)

results = db.similarity_search(query, k=1)
