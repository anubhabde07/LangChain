from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
from pydantic import BaseModel

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# Schema
class review(BaseModel):
    summary: str
    sentiment: str

stuctured_model = model.with_structured_output(review)

result = stuctured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other
brands. Hoping for a software update to fix this.""")

print(result)