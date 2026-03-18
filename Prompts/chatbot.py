from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-72B-Instruct",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

chat_history : list[BaseMessage] = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:
    user_input = input("Enter : ")
    if user_input == 'exit':
        break

    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI : ", result.content)

print(chat_history)