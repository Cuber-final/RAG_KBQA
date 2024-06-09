from langchain.tools import Tool
from agent.tools import *

tools = [
    Tool.from_function(
        func=calculate_math,
        name="calculate",
        description="Useful for when you need to answer questions about simple calculations",
        args_schema=CalculatorInput,
    ),
    Tool.from_function(
        func=search_ddgo,
        name="web_search",
        description="Useful for when you need to answer questions by internet",
        args_schema=SearchInternetInput,
    ),
    Tool.from_function(
        func=get_current_time,
        name="get_current_time",
        description="获取当前日期时间",
        args_schema=FuncWithOutInput,
    ),
]

tool_names = [tool.name for tool in tools]
