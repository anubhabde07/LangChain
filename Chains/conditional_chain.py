from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negetive'] = Field(description='Give the sentiment of the following feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the folowing feedback text into positive and negetive\n {feedback}\n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={
        'format_instructions': parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write a appropiate response for this positive feedback\n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write a appropiate response for this negetive feedback\n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negetive', prompt3 | model | parser),
    RunnableLambda(lambda x: "couldn't find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a fabulous gadget'})

print(result)

chain.get_graph().print_ascii()