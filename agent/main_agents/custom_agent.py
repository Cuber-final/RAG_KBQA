from agent.tools.calculate import PROMPT as calculate_prompt
from datetime import datetime
from langchain.tools import Tool
from langchain.agents import load_tools
from dotenv import load_dotenv, find_dotenv
import os
from langchain_zhipu import ChatZhipuAI
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
    create_openai_tools_agent,
)
from langchain.chains import LLMMathChain
from langchain.agents import load_tools
from langchain.agents import AgentType


def init_llm():
    load_dotenv(find_dotenv(), override=True)
    API_KEY = os.getenv("ZHIPUAI_API_KEY")
    llm = ChatZhipuAI(api_key=API_KEY, model="glm-4", verbose=True, temperature=0.8)
    return llm


def chat_agent():
    llm = init_llm()

    def get_current_time(query):
        return datetime.now()

    def calculate(query: str):
        llm_math = LLMMathChain.from_llm(llm, verbose=True, prompt=calculate_prompt)
        ans = llm_math.invoke(query)
        return ans

    tools = [
        Tool(
            name="get_current_time",
            func=get_current_time,
            description="获取当前日期时间",
        ),
        Tool(
            name="calculate",
            func=calculate,
            description="计算问题",
        ),
    ]

    # print(prompt)
    # agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # res = agent_executor.invoke({"input": "今天是几号？"})
    # print(res)

    # res = agent_executor.invoke({"input": "200*304等于多少？"})
    # print(res)

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    res = agent.invoke({"input": "今天是几号？"})
    print(res)
    # ares = agent.batch([{"input": "20*100-10 = ？"}, {"input": "今天是几号？"}])
    # 测试带历史记录的对话
    # print(ares)


if __name__ == "__main__":
    # agent_by_openai()
    # agent_by_chatglm()
    chat_agent()
    pass
