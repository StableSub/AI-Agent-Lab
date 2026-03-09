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

from langchain_chroma import Chroma

CHROMA_PATH = "./.chroma_db"

db = Chroma.from_documents(
    documents=recursive_docs,
    embedding=cached_embedder,
    persist_directory=CHROMA_PATH,
    collection_metadata={"hnsw:space": "cosine"} # cosine(코사인 유사도), l2(유클리드), ip(내적)
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4}
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

# 가장 비슷한 것만 찾으면 똑같은 내용의 문장만 N가지 나올 수 있음.
# MMR을 사용하면 관련 있으면서도 서로 다른 즉, 다양한 내용을 찾아 줌.