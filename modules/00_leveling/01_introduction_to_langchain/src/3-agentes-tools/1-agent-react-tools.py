from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from dotenv import load_dotenv

_ = load_dotenv()

@tool
def calculadora_expr(expression: str) -> str:
    """Evaluate a simple mathematical expression and returns the result"""
    try:
        print("----> calculator")
        result = eval(expression)
    except Exception as e:
        return f"Error: {e}"
    
    return str(result)

@tool
def web_search_fake(query: str) -> str:
    """Mocked web search tool. Return a hardcoded result."""
    data = {"Brazil": "Brasiliaaaa", "France": "Paris", "Germany": "Berlin", "Italy": "Rome"}

    print("----> web_search_mock")

    for country, capital in data.items():
        if country.lower() in query:
            return f"The capital of {country} is {capital}"
        
    return "I don't know the capital"

model = ChatOpenAI(model="gpt-5-mini", disable_streaming=True)
tools_to_use = [calculadora_expr, web_search_fake]


prompt = PromptTemplate.from_template(
"""
Answer the follwing question as best you can. You have access to de following tools.
Only use the information you get from the tools, even if you know the answer.
If the information is not provided by the tools, you say dont't know.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I know the final answer
Final Answer: the final answer to the original input question

Rules:
- If you choose an Action, do NOT include Final Answer in the same step.
- After Action and Action Input, stop and wait for the Observations.
- Never search the internet. Only use the tools provided.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)

agent_chain = create_react_agent(model, tools_to_use, prompt, stop_sequence=False)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain, 
    tools=tools_to_use, 
    verbose=True, 
    handle_parsing_errors="Invalid format. Either provide an Action with Action Input, or a Final Answer Only.", 
    max_interation=3)

#print(agent_executor.invoke({"input": "What is the captital of Brazil?"}))
print(agent_executor.invoke({"input": "What is the captital of Iran?"}))
#print(agent_executor.invoke({"input": "How much is 10 + 10?"}))