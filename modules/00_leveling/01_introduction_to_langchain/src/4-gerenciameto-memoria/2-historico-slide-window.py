from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from dotenv import load_dotenv

_ = load_dotenv()


def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get('raw_history', [])
    trimmed = trim_messages(
        raw_history, 
        token_counter=len, 
        max_tokens=2,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False
        )
    return {'input': payload.get('input', ''), 'history': trimmed}

prepare = RunnableLambda(prepare_inputs)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant'),
    MessagesPlaceholder(variable_name='history'),
    ('human', '{input}')
])

model = ChatOpenAI(model='gpt-5-nano', temperature=0.9, verbose=True)
chain = prepare | prompt | model

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain, 
    get_session_history,
    input_messages_key="input", 
    history_messages_key="raw_history"
)

config = {'configurable': {'session_id': 'demo-session'}}
inputs = [
    'Hello, my name is John. Reply with "OK" and do not mention my name',
    'Tell me a one-sentence fun fact. Do not mention my name.',
    'What is my name?'
]

for input in inputs:
    response = conversational_chain.invoke({"input": input}, config=config)
    print("Assistent: ", response.content)
    print("-"*30)

