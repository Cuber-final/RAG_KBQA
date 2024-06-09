from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from dotenv import find_dotenv, load_dotenv
import os
from langchain_zhipu import ChatZhipuAI
from models.model_contain import model_container
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS

# 使用搜索引擎检索答案并总结
# 搜索模板
request_template = """
在 >>> 和 <<< 之间的内容是网络搜索到的结果；请根据这部分结果抽取出关于问题{query}的准确答案，总结并回答用户问题。注意：用尽可能简洁清晰的中文回答。如果没有查找到答案，回答“抱歉，未找到答案"即可
>>> {requests_result} <<<
回答：
"""
PROMPT_REQUEST = PromptTemplate(
    input_variables=["query", "requests_result"], template=request_template
)


class SearchInternetInput(BaseModel):
    query: str = Field()


def search_ddgo(query: str):
    model = model_container.MODEL
    llm_chain = LLMChain(llm=model, prompt=PROMPT_REQUEST, verbose=True)

    # 使用DuckDuckGo进行搜索
    ddgs = DDGS(proxy=None, timeout=20)
    results = ddgs.text(query, max_results=2)
    web_content = ""
    if results:
        for result in results:
            web_content += result["body"]

    # 构造输入
    inputs = {
        "query": query,
        "requests_result": web_content,
    }

    # 运行链并获取结果
    answer = llm_chain.invoke(inputs)
    return answer


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    API_KEY = os.getenv("ZHIPUAI_API_KEY")

    llm = ChatZhipuAI(api_key=API_KEY, model="glm-4", temperature=0.8)
    model_container.MODEL = llm
    res = search_ddgo("海底捞是什么？")
    print(res["text"])
