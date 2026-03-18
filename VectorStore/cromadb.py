from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('virat_kohli.pdf')

docs = loader.load()

# print(len(docs))

# for i in range(len(docs)):
#     print(docs[i])


vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory='chroma_db',
    collection_name='sample'
)

# Add documents
vector_store.add_documents(docs)

# view documents
view_documents = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
# print(view_documents)

# search documents
# similar = vector_store.similarity_search(
#     query='Virat Kohli belongs to Punjab',
#     k=2
# )

# print(similar)


# search documents with score
similar_with_score = vector_store.similarity_search_with_score(
    query='Virat Kohli belongs to Punjab',
    k=1
)

print(similar_with_score)

# update documents
# updated_doc1 = Document(
# page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consister"
# metadata={"team": "Royal Challengers Bangalore"}
# )

# vector_store.update_document(document_id='fdb1dc2c-b27b-4f18-80ef-ee48f4d9a31f', document=updated_doc1)