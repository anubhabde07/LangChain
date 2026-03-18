
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever()

query = 'geoplitical history of china and india'
docs = retriever.invoke(query)


for i, doc in enumerate(docs):
    print(f'Result {i+1}')
    print(f'Content\n{doc.page_content}')