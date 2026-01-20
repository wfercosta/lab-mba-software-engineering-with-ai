from langchain_openai  import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

_ = load_dotenv()

long_text = """
Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.[34] Python is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming.

Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language. Python 3.0, released in 2008, was a major revision and not completely backward-compatible with earlier versions. Beginning with Python 3.5,[35] capabilities and keywords for typing were added to the language, allowing optional static typing.[36] As of 2026, the Python Software Foundation supports Python 3.10, 3.11, 3.12, 3.13, and 3.14, following the project's annual release cycle and five-year support policy. Earlier versions in the 3.x series have reached end-of-life and no longer receive security updates.

Python has gained widespread use in the machine learning community.[37][38][39][40] It is widely taught as an introductory programming language.[41] Since 2003, Python has consistently ranked in the top ten of the most popular programming languages in the TIOBE Programming Community Index, which ranks based on searches in 24 platforms.[42]
"""


spliter = RecursiveCharacterTextSplitter(
    chunk_size=250, 
    chunk_overlap=70
)

parts = spliter.create_documents([long_text])

# for part in parts:
#     print(part.page_content)
#     print("-"*10)

model = ChatOpenAI(model="gpt-5-nano", temperature=0)

map_prompt = PromptTemplate.from_template("Write a concise summary of the following text:\n{context}")
map_chain = map_prompt | model | StrOutputParser()

prepare_map_inputs = RunnableLambda(lambda docs: [{'context': d.page_content} for d in docs])
map_stage = prepare_map_inputs | map_chain.map()

reduce_prompt = PromptTemplate.from_template("Combine the following summaries into a sigla consice summary:\n{context}")
reduce_chain = reduce_prompt | model | StrOutputParser()

prepare_reduce_input = RunnableLambda(lambda summaries: {'context': '\n'.join(summaries)})
pipeline = map_stage | prepare_reduce_input | reduce_chain

result = pipeline.invoke(parts)
print(result)