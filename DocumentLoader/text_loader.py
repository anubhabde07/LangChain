from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary on the following text\n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('virat.txt', encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content})

print(result)