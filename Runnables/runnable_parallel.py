from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

llm1 = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation'
)

model1 = ChatHuggingFace(llm=llm1)

# llm2 = HuggingFaceEndpoint(
#     model='tiiuae/Falcon3-10B-Instruct',
#     task='text-generation'
# )

# model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template='Write a linkedin post about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write a twitter post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'linkedin': RunnableSequence(prompt1, model1, parser),
    'tweet': RunnableSequence(prompt2, model1, parser)
})

result = parallel_chain.invoke('Machine Learning')

print(result)