from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredPDFLoader

loader = DirectoryLoader(
    path='Books',
    glob='*.pdf',
    loader_cls=UnstructuredPDFLoader
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)