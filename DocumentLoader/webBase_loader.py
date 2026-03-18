from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Summarize the follwing website text\n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

url = 'https://en.wikipedia.org/wiki/LangChain'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})
print(result)