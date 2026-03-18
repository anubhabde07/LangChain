from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('virat_kohli.pdf')

docs = loader.load()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = FAISS.from_documents(docs, embedding)

vector_store.save_local('faiss_index')