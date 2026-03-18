from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

# Load chat_history
with open('chat_history.txt', 'r') as f:
    lines = f.readlines()

for line in lines:
    chat_history.append(line)

print(chat_history)

# Chat prompt
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'where is my refund'
})

print(prompt)