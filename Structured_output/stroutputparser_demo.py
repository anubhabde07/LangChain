from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# 1st prompt --> detailed report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt --> summarize
template2 = PromptTemplate(
    template="write a 5 line summary on the following text \n{text}",
    input_variables=['text']
)

prompt1 = template1.invoke({
    'topic': "video to text summarizaion using langchain"
})

result = model.invoke(prompt1)

prompt2 = template2.invoke({
    'text': result
})

result = model.invoke(prompt2)

print(result.content)