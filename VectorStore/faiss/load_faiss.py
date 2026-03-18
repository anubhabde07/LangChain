from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_store = FAISS.load_local(
    'faiss_index',
    embedding,
    allow_dangerous_deserialization=True
)

query = "Who is virat kohli"

result = vector_store.similarity_search(query, k=2)

for i, doc in enumerate(result):
    print(f"{i} : {doc.page_content}\n")