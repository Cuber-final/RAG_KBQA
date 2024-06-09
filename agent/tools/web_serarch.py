from langchain_community.utilities import BingSearchAPIWrapper
from configs.model_config import BING_SEARCH_URL, BING_SUBSCRIPTION_KEY
from langchain.chains import LLMRequestsChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_zhipu import ChatZhipuAI
from duckduckgo_search import DDGS
import dotenv
import os


# 没有APIkey，没信用卡申请暂未实现
def bing_search(text, result_len=3):
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [
            {
                "snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                "title": "env inof not fould",
                "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html",
            }
        ]
    search = BingSearchAPIWrapper(
        bing_subscription_key=BING_SUBSCRIPTION_KEY, bing_search_url=BING_SEARCH_URL
    )
    return search.results(text, result_len)


# 谷歌搜索
def google_search(query, llm):
    request_template = """
    在 >>> 和 <<< 之间的内容是谷歌搜索到的结果；请根据谷歌搜索到的结果抽取出关于问题{query}的准确答案，总结并回答用户问题。注意：用尽可能简洁清晰的中文回答。如果没有查找到答案，回答“抱歉，未找到答案"即可
    >>> {requests_result} <<<
    回答：
    """
    PROMPT_REQUEST = PromptTemplate(
        input_variables=["query", "requests_result"], template=request_template
    )

    llm_chain = LLMChain(llm=llm, prompt=PROMPT_REQUEST, verbose=True)

    chain = LLMRequestsChain(
        llm_chain=llm_chain,
        verbose=True,
    )

    inputs = {
        "query": query,
        "url": "https://www.google.com/search?q=" + query.replace(" ", "+"),  # 谷歌搜索
        # 'url':"https://cn.bing.com/search?q=" + question.replace(' ', '+')# 必应搜索
        # 'url' : "https://www.baidu.com/s?wd=" + question.replace(' ', '+')# 百度搜索
    }

    web_content = chain.run(inputs)
    # result_bing = chain.run(inputs) #必应的搜索不成功
    # result_baidu = chain.run(inputs) #百度可以，但是verbose输出的信息很杂乱
    # print(result_baidu)
    # print(result_google)
    return web_content


def ddgo_search(query, llm):
    # 搜索模板
    request_template = """
    在 >>> 和 <<< 之间的内容是网络搜索到的结果；请根据这部分结果抽取出关于问题{query}的准确答案，总结并回答用户问题。注意：用尽可能简洁清晰的中文回答。如果没有查找到答案，回答“抱歉，未找到答案"即可
    >>> {requests_result} <<<
    回答：
    """
    PROMPT_REQUEST = PromptTemplate(
        input_variables=["query", "requests_result"], template=request_template
    )

    # 创建LLMChain
    llm_chain = LLMChain(llm=llm, prompt=PROMPT_REQUEST, verbose=True)

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
    answer = llm_chain.run(inputs)
    return answer


if __name__ == "__main__":
    dotenv.load_dotenv()
    API_KEY = os.getenv("ZHIPUAI_API_KEY")
    llm = ChatZhipuAI(api_key=API_KEY, model="glm-4")
    # r = bing_search("python")

    # r = google_search("chatgpt-4o模型是什么？和GPT4有什么不同？", llm=llm)
    r = ddgo_search("chatgpt-4o模型是什么？和GPT4有什么不同？", llm=llm)

    # print(r)
