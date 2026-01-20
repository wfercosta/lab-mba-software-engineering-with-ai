from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

question_prompt = PromptTemplate(
    input_variables=["name"],
    template="Olá eu sou o {name}, conte uma piada com o meu nome"
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain = question_prompt | model

result = chain.invoke({"name": "The Rock"})

print(result.content)
