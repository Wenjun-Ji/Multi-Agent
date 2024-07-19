import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableLambda
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from typing import List, Union
from langchain.chains.llm_math.base import LLMMathChain
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from utils import create_rag_chain
from utils import create_vectorstore
from utils import load_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# Enable LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""

# llm
os.environ["IFLYTEK_SPARK_APP_ID"] = ""
os.environ["IFLYTEK_SPARK_API_KEY"] = ""
os.environ["IFLYTEK_SPARK_API_SECRET"] = ""
os.environ["IFLYTEK_SPARK_API_URL"] = ""
os.environ["IFLYTEK_SPARK_llm_DOMAIN"] = ""
from langchain_community.chat_models import ChatSparkLLM
llm = ChatSparkLLM(request_timeout=60)

# # Load documents
print("Loading documents...")
local_data_dir = r"..\data\local_files"
web_urls_file_path = r"..\data\web_urls.txt"
video_urls_file_path = r"..\data\video_urls.txt"
docs = load_docs(local_data_dir, web_urls_file_path,video_urls_file_path)
print(f"Loaded {len(docs)} documents.")

# Create vectorstore
print("Creating vectorstore...")
vectorstore = create_vectorstore(docs)
retriever = vectorstore.as_retriever()
print("Vectorstore created.")


# 创建日常对话工具
system_template = "你是一个善于聊天的机器人，人们都很愿意和你聊天"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{input}')
])
parser = StrOutputParser()
chat_chain = prompt_template | llm | parser
# result = chat_chain.invoke({"input" : "你好"})
# print(type(result))
# print(result)

chat_tool = Tool.from_function(
    name="Chat",
    func=chat_chain.invoke,
    description="Useful for having a chat with the AI. This tool is specifically designed for general conversation."
)


# 创建检索工具
rag_chain = create_rag_chain(llm, retriever)
# result = rag_chain.invoke({"input":"我想知道有关langchain开发的知识"})
# print(type(result))
# print(result)

rag_tool = Tool.from_function(
    name="Rag",
    func=rag_chain.invoke,
    description="Useful for answering questions about learning and developing with LangChain. This tool is specifically designed for handling queries related to LangChain tutorials, documentation, and best practices."
)

# 创建 DuckDuckGo 搜索工具和数学工具
duckduckgo_tool = Tool(
    name = "Search",
    func =DuckDuckGoSearchRun(),
    description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions."
)

problem_chain = LLMMathChain.from_llm(llm=llm)
# result = problem_chain.run("1+1=?")
# print(type(result))
# print(result)
math_tool = Tool.from_function(name="Calculator",
                 func=problem_chain.run,
                 description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions."
)



tools = [chat_tool,duckduckgo_tool,math_tool ,rag_tool]


# 定义提示模板
template_with_history = """
你是一个智能的对话机器人，你有以下工具：{tools}

其中Chat工具用来和用户完成日常对话，rag工具用来检索有关LangChain学习和开发的知识，Search工具用来搜索互联网以查找有关世界事件、问题、日期、年份等的信息，Calculator工具用来回答数学问题。这些工具并非一定使用
当你发现无法轻松的回答我的问题时，你可以使用以下工具：

{tools}


使用Rag、Search、Calculator工具时使用以下格式（注意Chat工具无需使用该格式）：

Question: 你必须回答的输入问题
Thought: 你应该一直思考要做什么
Action: 要采取的行动，应为以下之一 [{tool_names}]
Action Input: 行动的输入
Observation: 行动的结果
... (这个 Thought/Action/Action Input/Observation 部分可以重复 N 次)
Thought: 我现在知道最终答案了
Final Answer: 对原始输入问题的最终答案，请确保提供详细且丰富的回答。

开始吧！

Previous conversation history:
{history}

New question: {input}

{agent_scratchpad}

"""




class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        print(llm_output)
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                # Remind the user to ask a question
                return_values={"output": llm_output},
                log=llm_output,
            )
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()


# 创建代理和执行器
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser = output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

from langchain.memory import ConversationBufferWindowMemory
memory=ConversationBufferWindowMemory(k=2)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)


def parse_agent_output(agent_output):
    return agent_output["output"]

# result = agent_executor.invoke("How many people live in China as of 2024?")
# print(type(result))
# print(result)

# Create a chain to handle the input and parse the output
chain = (agent_executor | RunnableLambda(parse_agent_output)).with_types(input_type=str, output_type=str)

# result = chain.invoke("How many people live in China as of 2010?")
# print(type(result))
# print(result)
#
# result = chain.invoke("How many people live in China as of 2020?")
# print(type(result))
# print(result)
