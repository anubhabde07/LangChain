from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person", gt=18)
    city: str = Field(description="Name of the city belongs to the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate a name, age and city of a fictional {place} person \n {format_instructions}",
    input_variables=['place'],
    partial_variables={
        'format_instructions': parser.get_format_instructions()
    }
)

# prompt = template.invoke({'place':'Indian'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({'place': 'indian'})

print(final_result)