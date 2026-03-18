from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = HuggingFaceEmbeddings()

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name='my_collection'
)

retriever = vector_store.as_retriever()

query = 'What is Chroma used for'

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f'----Result {i+1}----')
    print(f'Content\n{doc.page_content}')