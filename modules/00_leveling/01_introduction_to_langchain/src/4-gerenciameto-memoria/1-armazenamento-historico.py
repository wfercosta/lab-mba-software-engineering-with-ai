from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv

_ = load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant'),
    MessagesPlaceholder(variable_name='history'),
    ('human', '{input}')
])

model = ChatOpenAI(model='gpt-5-nano', temperature=0.9, verbose=True)

chain = prompt | model

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain, 
    get_session_history,
    input_messages_key="input", 
    history_messages_key="history"
)

config = {'configurable': {'session_id': 'demo-session'}}
inputs = [
    'Hello, my name is John. How are you?',
    'Can you repeat my name?',
    'Can you repeat my name in a motivation phrase?'
]

for input in inputs:
    response = conversational_chain.invoke({"input": input}, config=config)
    print("Assistent: ", response.content)
    print("-"*30)