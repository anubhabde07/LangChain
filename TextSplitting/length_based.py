from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

loader = PyPDFLoader('MachineLearning.pdf')
docs = loader.load()

result = splitter.split_documents(docs)

print(len(result))
print(result[100].page_content)