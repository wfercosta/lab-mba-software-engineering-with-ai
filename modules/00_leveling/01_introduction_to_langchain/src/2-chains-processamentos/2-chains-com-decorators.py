from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()


@chain
def square(values: dict) -> dict:
    x = values["x"]
    return  {"square_result": x * x}


question_prompt = PromptTemplate(
    input_variables=["square_result"],
    template="Me conte alguma coisa sobre o número {square_result}"
)

model = ChatOpenAI(model="gpt-5-mini", temperature=0.5)

chain = square | question_prompt | model

result = chain.invoke({"x": 10})

print(result.content)
